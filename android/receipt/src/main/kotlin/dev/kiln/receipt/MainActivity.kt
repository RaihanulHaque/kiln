package dev.kiln.receipt

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.media.ExifInterface
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.view.Gravity
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import dev.kiln.ChatOptions
import dev.kiln.ContextParams
import dev.kiln.KilnContext
import dev.kiln.KilnModel
import dev.kiln.Media
import dev.kiln.Message
import dev.kiln.OcrOptions
import dev.kiln.OcrResult
import dev.kiln.Sampling
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.max

/**
 * Hishab Receipt, a temporary test app: a photo of a shop receipt becomes one Hishab ledger line.
 *
 *   photo --kiln OCR--> lines --rules (Receipt.kt)--> "Daily Fresh grocery 330" --Hishab model--> one line
 *
 * Two models, both pushed into the app's folder with adb (see README.md). Everything kiln does
 * runs on one background thread ([worker]); views are touched on the main thread only.
 */
class MainActivity : Activity() {

    private val worker = Executors.newSingleThreadExecutor()
    private val ui = Handler(Looper.getMainLooper())

    // only touched on the worker thread
    private var ocrModel: KilnModel? = null
    private var ocrCtx: KilnContext? = null
    private var hishabModel: KilnModel? = null
    private var hishabCtx: KilnContext? = null

    private lateinit var modelLine: TextView
    private lateinit var statusLine: TextView
    private lateinit var card: LinearLayout
    private lateinit var image: ImageView
    private lateinit var rowsOut: LinearLayout
    private lateinit var buttons: List<Button>

    private var sample = 0

    private class Page(val name: String, val bytes: ByteArray, val preview: Bitmap, val scale: Float)

    private class Outcome(
        val page: Page,
        val ocr: OcrResult,
        val receipt: Receipt.Result,
        val answer: String?,
        val hishabMs: Double,
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildViews()
        worker.execute { loadModels() }
        if (intent?.action == Intent.ACTION_SEND) handleShare(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        if (intent.action == Intent.ACTION_SEND) handleShare(intent)
    }

    override fun onDestroy() {
        super.onDestroy()
        worker.execute {
            ocrCtx?.close(); ocrModel?.close()
            hishabCtx?.close(); hishabModel?.close()
        }
        worker.shutdown()
    }

    // ---- models ------------------------------------------------------------------------------

    private val ocrName = "ocria-nano-1.gguf"

    private fun modelDir(): File? = getExternalFilesDir(null)

    /** The Hishab model: any .gguf in the app folder whose name says hishab or pipe_signed. */
    private fun hishabFile(): File? = modelDir()?.listFiles()?.firstOrNull {
        it.isFile && it.name.endsWith(".gguf") && (it.name.contains("hishab") || it.name.contains("pipe_signed"))
    }

    private fun loadModels() {
        val dir = modelDir()?.absolutePath ?: "(no external storage)"
        val status = StringBuilder()
        try {
            val f = File(modelDir(), ocrName)
            if (f.isFile) {
                val t0 = System.nanoTime()
                ocrModel = KilnModel.load(f.absolutePath)
                ocrCtx = ocrModel!!.newContext()
                status.append(String.format(Locale.US, "OCR: %s (%.0f ms)\n", f.name, (System.nanoTime() - t0) / 1e6))
            } else {
                status.append("OCR model missing: adb push models/$ocrName $dir/\n")
            }
            val h = hishabFile()
            if (h != null) {
                val t0 = System.nanoTime()
                hishabModel = KilnModel.load(h.absolutePath)
                hishabCtx = hishabModel!!.newContext(ContextParams(nCtx = 2048))
                status.append(String.format(Locale.US, "Hishab: %s (%.0f ms, %s)", h.name, (System.nanoTime() - t0) / 1e6, hishabModel!!.info.backendName))
            } else {
                status.append("Hishab model missing: adb push <the L20 .gguf> $dir/")
            }
        } catch (e: Exception) {
            status.append("\nCould not load: ${e.message}")
        }
        post {
            modelLine.text = status.toString().trim()
            setBusy(false)
        }
    }

    // ---- the pipeline ------------------------------------------------------------------------

    /** The exact system prompt Hishab was trained with (ml/hishab/prompts.py, pipe_signed). */
    private val system =
        "You are Hishab, a Bangladeshi personal-finance assistant. The user writes in English, " +
            "Bangla or Banglish (Bangla in Latin letters), often mixed. Extract every financial " +
            "transaction they mention, one per line as description|amount|category. amount is a BDT " +
            "number (expand shorthand like 1.5k or 2 lakh) written with - for an expense and + for income. " +
            "category is one of: food, grocery, rent, utilities, mobile_recharge, transport, tuition, " +
            "salary, gift, loan, shopping, medical, entertainment, other. " +
            "If there is no transaction, reply NONE. Reply with the lines only."

    private fun read(name: String, load: () -> ByteArray) {
        setBusy(true)
        statusLine.text = "Reading the receipt…"
        worker.execute {
            try {
                val page = preparePage(name, load())
                val c = ocrCtx ?: error("the OCR model is not loaded")
                val ocr = c.ocrCollect(Media.Bytes(page.bytes), OcrOptions())
                val receipt = Receipt.read(ocr.lines)
                var answer: String? = null
                var ms = 0.0
                val h = hishabCtx
                if (receipt.message != null && h != null) {
                    post { statusLine.text = "Asking Hishab: ${receipt.message}" }
                    val t0 = System.nanoTime()
                    val reply = h.chatCollect(
                        listOf(Message.system(system), Message.user(receipt.message)),
                        ChatOptions(enableThinking = false),
                        Sampling.greedy(maxTokens = 512),
                    )
                    ms = (System.nanoTime() - t0) / 1e6
                    answer = reply.answer.trim()
                }
                val out = Outcome(page, ocr, receipt, answer, ms)
                post { show(out) }
            } catch (e: Throwable) {
                post {
                    statusLine.text = "Failed: ${e.message}"
                    setBusy(false)
                }
            }
        }
    }

    // ---- showing the result ------------------------------------------------------------------

    private fun show(o: Outcome) {
        val r = o.receipt
        statusLine.text = String.format(
            Locale.US, "%s · OCR %.0f ms (%d lines) · Hishab %.0f ms · tilt %.1f°",
            o.page.name, o.ocr.stats.ms, o.ocr.stats.lines, o.hishabMs, r.tiltDegrees,
        )
        card.removeAllViews()
        val entries = o.answer?.let { Receipt.parseAnswer(it) } ?: emptyList()
        if (entries.size == 1) {
            val e = entries[0]
            card.addView(text("৳ ${Receipt.fmt(-e.amount)}   ${e.category}", 30f, bold = true))
            card.addView(text(e.description, 18f))
        } else if (o.answer != null) {
            card.addView(text("Hishab gave ${entries.size} lines", 22f, bold = true))
        } else {
            card.addView(text("No total found", 24f, bold = true))
        }
        val (label, color) = when (r.status) {
            Receipt.Status.SURE -> "✓ Total checked" to Color.rgb(0x18, 0x80, 0x38)
            Receipt.Status.FIXED -> "✓ Total label unreadable, worked out from the other lines" to Color.rgb(0x18, 0x80, 0x38)
            Receipt.Status.UNCHECKED -> "• Total not checked (nothing on the receipt to check it with)" to Color.rgb(0x80, 0x80, 0x80)
            Receipt.Status.CHECK -> "⚠ Please check the total: the numbers do not add up" to Color.rgb(0xd0, 0x70, 0x00)
            Receipt.Status.NONE -> "⚠ Could not find a total" to Color.rgb(0xc0, 0x30, 0x30)
        }
        card.addView(text(label, 15f, bold = true).apply { setTextColor(color); setPadding(0, dp(8), 0, dp(4)) })
        card.addView(small("Shop: ${r.shop ?: "?"}    Kind: ${r.kind ?: "?"}    Total line: ${r.total?.let { Receipt.fmt(it) } ?: "?"}"))
        r.checks.forEach { card.addView(small("Check: $it")) }
        card.addView(small("Sent to Hishab:  ${r.message ?: "(nothing)"}"))
        card.addView(small("Hishab answered:  ${o.answer ?: "(not asked)"}").apply { typeface = Typeface.MONOSPACE })

        drawPage(o)
        rowsOut.removeAllViews()
        r.rows.forEach { row ->
            val mark = when (row) {
                r.totalRow -> "TOTAL "
                r.shopRow -> "SHOP  "
                else -> "      "
            }
            rowsOut.addView(TextView(this).apply {
                text = String.format(Locale.US, "%s%.2f  %s", mark, row.conf, row.text)
                textSize = 13f
                typeface = Typeface.MONOSPACE
                setTextIsSelectable(true)
                setTextColor(
                    when (row) {
                        r.totalRow -> Color.rgb(0x18, 0x80, 0x38)
                        r.shopRow -> Color.rgb(0x1a, 0x73, 0xe8)
                        else -> if (row.conf < 0.6f) Color.rgb(0xb0, 0x30, 0x30) else Color.DKGRAY
                    },
                )
            })
        }
        setBusy(false)
    }

    private fun drawPage(o: Outcome) {
        val bmp = o.page.preview.copy(Bitmap.Config.ARGB_8888, true)
        val c = Canvas(bmp)
        val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = max(2f, bmp.width / 300f) }
        val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
        val k = o.page.scale
        for (row in o.receipt.rows) {
            val color = when (row) {
                o.receipt.totalRow -> Color.rgb(0x18, 0x80, 0x38)
                o.receipt.shopRow -> Color.rgb(0x1a, 0x73, 0xe8)
                else -> Color.rgb(0x90, 0x90, 0x90)
            }
            for (line in row.boxes) {
                val q = line.quad
                val path = Path().apply {
                    moveTo(q[0] * k, q[1] * k); lineTo(q[2] * k, q[3] * k)
                    lineTo(q[4] * k, q[5] * k); lineTo(q[6] * k, q[7] * k); close()
                }
                fill.color = (color and 0x00ffffff) or 0x30000000
                stroke.color = color
                c.drawPath(path, fill)
                c.drawPath(path, stroke)
            }
        }
        image.setImageBitmap(bmp)
    }

    // ---- pictures (from Lekha) ---------------------------------------------------------------

    /** kiln does not apply a photo's EXIF orientation: turn such photos upright first. */
    private fun preparePage(name: String, bytes: ByteArray): Page {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
        require(bounds.outWidth > 0) { "not an image kiln or Android can open" }
        val orientation = try {
            ExifInterface(bytes.inputStream()).getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        } catch (_: Exception) {
            ExifInterface.ORIENTATION_NORMAL
        }
        val turn = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> turn.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> turn.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> turn.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> turn.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> turn.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> { turn.postRotate(90f); turn.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_TRANSVERSE -> { turn.postRotate(270f); turn.postScale(-1f, 1f) }
        }
        if (turn.isIdentity) {
            val preview = decodeAtMost(bytes, bounds.outWidth, bounds.outHeight, 2048)
            return Page(name, bytes, preview, preview.width.toFloat() / bounds.outWidth)
        }
        val src = decodeAtMost(bytes, bounds.outWidth, bounds.outHeight, 3200)
        val upright = Bitmap.createBitmap(src, 0, 0, src.width, src.height, turn, true)
        val jpeg = ByteArrayOutputStream().also { upright.compress(Bitmap.CompressFormat.JPEG, 95, it) }.toByteArray()
        val preview = if (max(upright.width, upright.height) > 2048) {
            val s = 2048f / max(upright.width, upright.height)
            Bitmap.createScaledBitmap(upright, (upright.width * s).toInt(), (upright.height * s).toInt(), true)
        } else upright
        return Page(name, jpeg, preview, preview.width.toFloat() / upright.width)
    }

    private fun decodeAtMost(bytes: ByteArray, w: Int, h: Int, longest: Int): Bitmap {
        var sample = 1
        while (max(w, h) / (sample * 2) >= longest) sample *= 2
        val b = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, BitmapFactory.Options().apply { inSampleSize = sample })
        val long = max(b.width, b.height)
        if (long <= longest) return b
        val s = longest.toFloat() / long
        return Bitmap.createScaledBitmap(b, (b.width * s).toInt().coerceAtLeast(1), (b.height * s).toInt().coerceAtLeast(1), true)
    }

    private val pickImage = 1
    private val takePhoto = 2
    private val photoFile by lazy { PhotoProvider.file(this) }

    private fun openGallery() {
        try {
            startActivityForResult(Intent(MediaStore.ACTION_PICK_IMAGES).setType("image/*"), pickImage)
        } catch (_: Exception) {
            startActivityForResult(Intent(Intent.ACTION_GET_CONTENT).setType("image/*"), pickImage)
        }
    }

    private fun openCamera() {
        photoFile.delete()
        val i = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            .putExtra(MediaStore.EXTRA_OUTPUT, PhotoProvider.uri)
            .addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION or Intent.FLAG_GRANT_READ_URI_PERMISSION)
        try {
            startActivityForResult(i, takePhoto)
        } catch (_: Exception) {
            toast("No camera app found")
        }
    }

    private fun openSample() {
        val names = assets.list("")!!.filter { it.endsWith(".jpg") }.sorted()
        val name = names[sample++ % names.size]
        read("sample $name") { assets.open(name).use { it.readBytes() } }
    }

    private fun handleShare(intent: Intent) {
        @Suppress("DEPRECATION")
        val uri = intent.getParcelableExtra<Uri>(Intent.EXTRA_STREAM) ?: return
        read("shared picture") { readUri(uri) }
    }

    private fun readUri(uri: Uri): ByteArray = contentResolver.openInputStream(uri)!!.use { it.readBytes() }

    @Deprecated("startActivityForResult keeps this app free of androidx.activity")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        @Suppress("DEPRECATION")
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode != RESULT_OK) return
        when (requestCode) {
            pickImage -> data?.data?.let { uri -> read("gallery picture") { readUri(uri) } }
            takePhoto -> read("camera photo") { photoFile.readBytes() }
        }
    }

    // ---- views -------------------------------------------------------------------------------

    private fun buildViews() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(16), dp(16), dp(24))
        }
        root.addView(text("Hishab Receipt", 28f, bold = true))
        root.addView(small("Photo of a receipt → one Hishab entry with the final total. Test app; nothing leaves the phone."))
        modelLine = small("Loading the models…").apply { setPadding(0, dp(8), 0, dp(8)) }
        root.addView(modelLine)

        val camera = button("Camera") { openCamera() }
        val gallery = button("Gallery") { openGallery() }
        val sampleB = button("Sample") { openSample() }
        root.addView(LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            listOf(camera, gallery, sampleB).forEach { addView(it, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)) }
        })
        buttons = listOf(camera, gallery, sampleB)

        statusLine = small("Take a photo of a shop receipt, pick one, or try a sample.").apply {
            setTextColor(Color.BLACK); setPadding(0, dp(12), 0, dp(8))
        }
        root.addView(statusLine)
        card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(12), dp(14), dp(12))
            background = GradientDrawable().apply {
                setColor(Color.rgb(0xf4, 0xf6, 0xf8)); cornerRadius = dp(10).toFloat()
            }
        }
        root.addView(card, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(header("The receipt  (blue: shop, green: total)"))
        image = ImageView(this).apply {
            adjustViewBounds = true
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(Color.rgb(0xee, 0xee, 0xee))
        }
        root.addView(image, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(header("What the rules saw  (rows, lowest confidence, text)"))
        rowsOut = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        root.addView(rowsOut)

        setContentView(ScrollView(this).apply {
            setBackgroundColor(Color.WHITE)
            fitsSystemWindows = true
            addView(root)
        })
        setBusy(true)
    }

    private fun text(t: String, size: Float, bold: Boolean = false) = TextView(this).apply {
        text = t
        textSize = size
        setTextColor(Color.BLACK)
        setTextIsSelectable(true)
        if (bold) typeface = Typeface.DEFAULT_BOLD
    }

    private fun small(t: String) = TextView(this).apply {
        text = t
        textSize = 13f
        setTextIsSelectable(true)
        setPadding(0, dp(2), 0, dp(2))
    }

    private fun header(t: String) = TextView(this).apply {
        text = t
        textSize = 13f
        typeface = Typeface.DEFAULT_BOLD
        setPadding(0, dp(16), 0, dp(4))
    }

    private fun button(label: String, onClick: () -> Unit) = Button(this).apply {
        text = label
        isAllCaps = false
        setOnClickListener { onClick() }
    }

    private fun setBusy(busy: Boolean) = buttons.forEach { it.isEnabled = !busy }
    private fun post(block: () -> Unit) = ui.post(block)
    private fun toast(t: String) = Toast.makeText(this, t, Toast.LENGTH_SHORT).show()
    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()
}
