package dev.kiln.receipt

import dev.kiln.OcrLine
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

/**
 * The rules that turn OCR lines into one Hishab message. A port of run.py in
 * hishab/experiments/receipt_ocr; no model is involved here, only plain code:
 *
 *  1. undo the page tilt and join boxes at the same height into rows
 *  2. find the total: the row with the most specific label (GRAND TOTAL ... Total, সর্বমোট ... মোট)
 *  3. check it: subtotal (or the item prices) + VAT + service - discount, and cash paid - change
 *  4. the shop: the first line of letters at the top; the kind: count clue words (mg, biryani, ...)
 *  5. the message: "<shop> <kind> <total>"
 */
object Receipt {

    class Row(val y: Float, val boxes: List<OcrLine>) {
        val text: String = boxes.joinToString("  ") { it.text }
        val conf: Float = boxes.minOf { it.confidence }
    }

    enum class Status { SURE, FIXED, UNCHECKED, CHECK, NONE }

    class Result(
        val rows: List<Row>,
        val tiltDegrees: Double,
        val shop: String?,
        val shopRow: Row?,
        val kind: String?,
        val total: Double?,
        val totalRow: Row?,
        val status: Status,
        val checks: List<String>,
        val message: String?,
    )

    fun read(lines: List<OcrLine>): Result {
        val (rows, tilt) = rows(lines)
        val (label, totalRow) = findTotal(rows)
        val checks = mutableListOf<Pair<String, Double>>()
        var sub = amountAfter(rows, listOf("subtotal", "sub total"))
        var subName = "subtotal"
        if (sub == null) {
            sub = itemsSum(rows, totalRow)
            subName = "items"
        }
        if (sub != null) {
            val vat = amountAfter(rows, listOf("vat"), listOf("bin")) ?: 0.0
            val svc = amountAfter(rows, listOf("service")) ?: 0.0
            val disc = amountAfter(rows, listOf("discount", "ছাড়")) ?: 0.0
            val parts = buildString {
                append("$subName ${fmt(sub)}")
                if (vat > 0) append(" + VAT ${fmt(vat)}")
                if (svc > 0) append(" + service ${fmt(svc)}")
                if (disc > 0) append(" − discount ${fmt(disc)}")
            }
            checks += parts to round2(sub + vat + svc - disc)
        }
        val paid = amountAfter(rows, listOf("cash", "paid", "নগদ"), listOf("card", "bkash"))
        val change = amountAfter(rows, listOf("change", "ফেরত"))
        if (paid != null && change != null) checks += "paid ${fmt(paid)} − change ${fmt(change)}" to round2(paid - change)

        val values = checks.map { it.second }
        val (total, status) = when {
            label != null && values.any { abs(label - it) < 0.01 } -> label to Status.SURE
            values.isNotEmpty() && values.all { abs(it - values[0]) < 0.01 } && (label == null || values.size == 2) ->
                values[0] to (if (values.size == 2) Status.SURE else Status.FIXED)
            label != null && values.isEmpty() -> label to Status.UNCHECKED
            label != null -> label to Status.CHECK
            else -> null to Status.NONE
        }
        val (shop, shopRow) = findShop(rows)
        val kind = findKind(rows)
        val message = total?.let { t ->
            val s = shop ?: "shop"
            if (kind != null) "$s $kind ${fmt(t)}" else "$s ${fmt(t)}"
        }
        return Result(
            rows, tilt, shop, shopRow, kind, total, totalRow, status,
            checks.map { (how, v) -> "$how = ${fmt(v)}" }, message,
        )
    }

    // ---- rows ---------------------------------------------------------------------------------

    private fun rows(lines: List<OcrLine>): Pair<List<Row>, Double> {
        if (lines.isEmpty()) return emptyList<Row>() to 0.0
        val angles = lines.mapNotNull { l ->
            val q = l.quad
            val w = q[2] - q[0]
            if (w > 80) atan2((q[3] - q[1]).toDouble(), w.toDouble()) else null
        }
        val ang = median(angles) ?: 0.0
        val c = cos(-ang)
        val s = sin(-ang)
        class P(val line: OcrLine, val rx: Double, val ry: Double, val h: Double)
        val ps = lines.map { l ->
            val q = l.quad
            val cx = (q[0] + q[2] + q[4] + q[6]) / 4.0
            val cy = (q[1] + q[3] + q[5] + q[7]) / 4.0
            val ys = listOf(q[1], q[3], q[5], q[7])
            P(l, cx * c - cy * s, cx * s + cy * c, (ys.max() - ys.min()).toDouble())
        }
        val tol = 0.45 * (median(ps.map { it.h * abs(cos(ang)) }) ?: 20.0)
        val groups = mutableListOf<MutableList<P>>()
        val ys = mutableListOf<Double>()
        for (p in ps.sortedBy { it.ry }) {
            if (groups.isNotEmpty() && abs(ys.last() - p.ry) < tol) {
                groups.last() += p
                ys[ys.size - 1] = groups.last().map { it.ry }.average()
            } else {
                groups += mutableListOf(p)
                ys += p.ry
            }
        }
        val rows = groups.mapIndexed { i, g -> Row(ys[i].toFloat(), g.sortedBy { it.rx }.map { it.line }) }
        return rows to Math.toDegrees(ang)
    }

    private fun median(xs: List<Double>): Double? {
        if (xs.isEmpty()) return null
        val s = xs.sorted()
        return if (s.size % 2 == 1) s[s.size / 2] else (s[s.size / 2 - 1] + s[s.size / 2]) / 2
    }

    // ---- numbers ------------------------------------------------------------------------------

    /** Look-alikes the world recognizer gives for faint Bangla digits. */
    private val confusable = mapOf('ა' to "1", 'ం' to "0", 'ை' to "")

    fun asciiDigits(s: String): String = buildString {
        for (ch in s) {
            val sub = confusable[ch]
            when {
                sub != null -> append(sub)
                Character.isDigit(ch) -> append(Character.digit(ch, 10))
                else -> append(ch)
            }
        }
    }

    private val decimalComma = Regex("(\\d),(\\d{2})(?!\\d)")
    private val lostDigit = Regex("(?<!\\d)[,.](\\d{3})")
    private val num = Regex(
        "(?<![\\w.#])-?\\d{1,3}(?:,\\d{3})+(?:\\.\\d{1,2})?(?!\\d)|(?<![\\w.,#])-?\\d+(?:\\.\\d{1,2})?(?![\\d,])",
    )
    private val hasLetter = Regex("[^\\W\\d_]")

    fun numbers(text: String): List<Double> {
        var t = asciiDigits(text)
        t = decimalComma.replace(t, "$1.$2")   // ৭০৫,০০ : a decimal point read as a comma
        t = lostDigit.replace(t, "#$1")        // ,530.00 : the first digit was lost; not a number
        return num.findAll(t).mapNotNull { it.value.replace(",", "").toDoubleOrNull() }.toList()
    }

    // ---- total --------------------------------------------------------------------------------

    private val totalKeys = listOf(
        "grand total" to 9, "net payable" to 9, "amount payable" to 9, "total payable" to 9,
        "net total" to 8, "total amount" to 8, "total due" to 8, "সর্বমোট" to 9, "নিট প্রদেয" to 9,
        "মোট প্রদেয" to 9, "মোট টাকা" to 8, "প্রদেয" to 7, "total" to 5, "মোট" to 5, "নিট" to 4,
    )
    private val notTotal = listOf("sub", "total qty", "total item", "items")

    private fun findTotal(rows: List<Row>): Pair<Double?, Row?> {
        var best: Pair<Int, Int>? = null
        rows.forEachIndexed { i, r ->
            val low = asciiDigits(r.text).lowercase()
            if (notTotal.any { it in low }) return@forEachIndexed
            val hit = totalKeys.firstOrNull { it.first in low } ?: return@forEachIndexed
            if (best == null || hit.second >= best!!.first) best = hit.second to i
        }
        val i = best?.second ?: return null to null
        numbers(rows[i].text).filter { it > 0 }.lastOrNull()?.let { return it to rows[i] }
        for (j in listOf(i - 1, i + 1)) {
            if (j !in rows.indices) continue
            val t = rows[j].text
            val ns = numbers(t)
            if (ns.isNotEmpty() && !hasLetter.containsMatchIn(asciiDigits(t))) return ns.last() to rows[j]
        }
        return null to rows[i]
    }

    private fun amountAfter(rows: List<Row>, keys: List<String>, exclude: List<String> = emptyList()): Double? {
        for (r in rows) {
            val low = r.text.lowercase()
            if (keys.any { it in low } && exclude.none { it in low }) {
                numbers(r.text).lastOrNull()?.let { return abs(it) }
            }
        }
        return null
    }

    private val itemSkip = listOf(
        "invoice", "date", "phone", "mobile", "bin", "table", "rate", "volume", "pump", "vehicle", "product",
        "মোবাইল", "রসিদ", "তারিখ", "তারিথ", "তারিয", "vat", "service", "discount", "ছাড়",
    )

    private fun itemsSum(rows: List<Row>, totalRow: Row?): Double? {
        val end = totalRow?.let { rows.indexOf(it) }?.takeIf { it >= 0 } ?: rows.size
        var s = 0.0
        var n = 0
        for (r in rows.subList(0, end)) {
            val low = r.text.lowercase()
            if (itemSkip.any { it in low }) continue
            val ns = numbers(r.text)
            if (ns.isNotEmpty() && hasLetter.containsMatchIn(asciiDigits(r.text))) {
                s += ns.last()
                n++
            }
        }
        return if (n > 0) round2(s) else null
    }

    // ---- shop and kind ------------------------------------------------------------------------

    private val junk = listOf(
        "invoice", "receipt", "bin", "vat", "phone", "mobile", "tel", "date", "মোবাইল", "রসিদ",
        "তারিখ", "road", "house", "dhaka", "ঢাকা",
    )

    private fun findShop(rows: List<Row>): Pair<String?, Row?> {
        for (r in rows.take(4)) {
            val t = r.text.trim()
            val low = t.lowercase()
            val letters = t.count { Character.isLetter(it) || Character.getType(it).let { ty ->
                ty == Character.NON_SPACING_MARK.toInt() || ty == Character.COMBINING_SPACING_MARK.toInt() } }
            val nonSpace = t.count { !it.isWhitespace() }.coerceAtLeast(1)
            if (letters < 3 || letters.toDouble() / nonSpace < 0.7) continue
            if (junk.any { it in low } || asciiDigits(t).any { it.isDigit() }) continue
            val name = if (t.all { it.code < 128 }) t.lowercase().split(" ").joinToString(" ") { w ->
                w.replaceFirstChar { it.uppercase() } } else t
            return name to r
        }
        return null to null
    }

    private val kinds = mapOf(
        "pharmacy" to listOf("pharmacy", "pharma", "drug", "mg", "syrup", "tab", "napa", "seclo", "orsaline",
            "ফার্মেসি", "ঔষধ", "সিরাপ", "নাপা", "সেকলো"),
        "restaurant" to listOf("restaurant", "cafe", "dhaba", "biryani", "kacchi", "tehari", "borhani", "table",
            "server", "service charge", "রেস্তো", "বিরিয়ানি", "বোরহানি", "রোস্ট", "ভাত"),
        "fuel" to listOf("filling station", "fuel", "petrol", "octane", "diesel", "pump", "nozzle", "tk/l"),
        "grocery" to listOf("super shop", "mart", "rice", "dal", "oil", "eggs", "sugar", "salt", "potato",
            "onion", "সুপার শপ", "চাল", "ডাল", "তেল", "ডিম", "আলু"),
        "clothing" to listOf("fashion", "shirt", "pant", "panjabi", "saree", "বস্ত্র", "শার্ট", "পাঞ্জাবি", "শাড়ি", "প্যান্ট"),
        "electronics" to listOf("tech", "cable", "charger", "power bank", "earphone", "phone cover"),
    )

    private fun findKind(rows: List<Row>): String? {
        val text = rows.joinToString(" ") { it.text }.lowercase()
        val scores = kinds.mapValues { (_, ws) -> ws.sumOf { w -> text.windowed(w.length).count { it == w } } }
        val best = scores.maxBy { it.value }
        return if (best.value > 0) best.key else null
    }

    // ---- Hishab's answer ----------------------------------------------------------------------

    class Entry(val description: String, val amount: Double, val category: String)

    fun parseAnswer(answer: String): List<Entry> = answer.lines().mapNotNull { line ->
        val bits = line.trim().split("|")
        if (bits.size != 3) return@mapNotNull null
        bits[1].trim().toDoubleOrNull()?.let { Entry(bits[0].trim(), it, bits[2].trim()) }
    }

    fun fmt(x: Double): String = if (x == Math.floor(x)) x.toLong().toString() else String.format(java.util.Locale.US, "%.2f", x)
    private fun round2(x: Double) = Math.round(x * 100) / 100.0
}
