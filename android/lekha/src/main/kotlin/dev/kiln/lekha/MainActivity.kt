package dev.kiln.lekha

import android.app.Activity
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.graphics.Typeface
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
import dev.kiln.KilnContext
import dev.kiln.KilnException
import dev.kiln.KilnModel
import dev.kiln.Media
import dev.kiln.OcrLine
import dev.kiln.OcrOptions
import dev.kiln.OcrResult
import dev.kiln.Script
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.math.max

/**
 * Lekha ("writing" in Bengali): take a photo, pick one, or share one into the app, and see what
 * kiln's OCR reads, where it found each line, and how long it took.
 *
 * Everything kiln does runs on one background thread ([worker]); the views are touched on the
 * main thread only ([ui]).
 */
class MainActivity : Activity() {

    private val worker = Executors.newSingleThreadExecutor()
    private val ui = Handler(Looper.getMainLooper())

    // only touched on the worker thread
    private var model: KilnModel? = null
    private var ctx: KilnContext? = null
    private var lastPage: Page? = null

    private lateinit var modelLine: TextView
    private lateinit var statsLine: TextView
    private lateinit var image: ImageView
    private lateinit var textOut: TextView
    private lateinit var linesOut: LinearLayout
    private lateinit var buttons: List<Button>
    private lateinit var longSideButton: Button
    private lateinit var boxesButton: Button

    private var longSide = 0          // 0 = kiln's default (1600)
    private var showBoxes = true
    private var sample = 0
    private var shown: Shown? = null  // what is on screen, to redraw when boxes are toggled

    /** A page ready for kiln: the bytes it reads, and a smaller picture to draw on. */
    private class Page(val name: String, val bytes: ByteArray, val preview: Bitmap, val scale: Float)

    private class Shown(val page: Page, val result: OcrResult)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        buildViews()
        worker.execute { loadModel() }
        if (intent?.action == Intent.ACTION_SEND) handleShare(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        if (intent.action == Intent.ACTION_SEND) handleShare(intent)
    }

    override fun onDestroy() {
        super.onDestroy()
        worker.execute {
            ctx?.close()
            model?.close()
        }
        worker.shutdown()
    }

    // ---- the model -------------------------------------------------------------------------

    private val modelName = "ocria-nano-1.gguf"

    /** Where the app looks for the model, in order. `adb push` can write to the second one. */
    private fun modelCandidates(): List<File> =
        listOfNotNull(File(filesDir, modelName), getExternalFilesDir(null)?.let { File(it, modelName) })

    private fun loadModel() {
        val file = modelCandidates().firstOrNull { it.isFile }
        if (file == null) {
            val ext = getExternalFilesDir(null)?.absolutePath ?: "(no external storage)"
            post {
                modelLine.text = "No model yet. Tap \"Model file\" and choose $modelName, or run:\n" +
                    "adb push models/$modelName $ext/"
                setBusy(false)
            }
            return
        }
        try {
            ctx?.close()
            model?.close()
            val t0 = System.nanoTime()
            val m = KilnModel.load(file.absolutePath)
            if (!m.info.isOcrModel) {
                m.close()
                post { modelLine.text = "${file.name} is not an OCR model (it is ${m.info.arch})." }
                return
            }
            model = m
            ctx = m.newContext()
            val ms = (System.nanoTime() - t0) / 1e6
            val i = m.info
            post {
                modelLine.text = String.format(
                    Locale.US, "%s · %.1f M parameters · %s · loaded in %.0f ms",
                    file.name, i.nParams / 1e6, i.backendName, ms,
                )
                setBusy(false)
            }
        } catch (e: KilnException) {
            post { modelLine.text = "Could not load ${file.name}: ${e.message}" }
        }
    }

    /** Copies a model file the user picked into the app's own folder, then loads it. */
    private fun importModel(uri: Uri) {
        setBusy(true)
        modelLine.text = "Copying the model…"
        worker.execute {
            try {
                val dst = File(filesDir, modelName)
                val tmp = File(filesDir, "$modelName.part")
                contentResolver.openInputStream(uri)!!.use { inp -> tmp.outputStream().use { inp.copyTo(it) } }
                tmp.renameTo(dst)
                loadModel()
            } catch (e: Exception) {
                post {
                    modelLine.text = "Could not copy the model: ${e.message}"
                    setBusy(false)
                }
            }
        }
    }

    // ---- reading a page --------------------------------------------------------------------

    private fun read(name: String, load: () -> ByteArray) {
        setBusy(true)
        statsLine.text = "Reading…"
        worker.execute {
            try {
                val page = preparePage(name, load())
                lastPage = page
                runOcr(page)
            } catch (e: Exception) {
                post {
                    statsLine.text = "Could not read the picture: ${e.message}"
                    setBusy(false)
                }
            }
        }
    }

    private fun runOcr(page: Page) {
        val c = ctx
        if (c == null) {
            post {
                statsLine.text = "Load the model first."
                setBusy(false)
            }
            return
        }
        try {
            val result = c.ocrCollect(Media.Bytes(page.bytes), OcrOptions(longSide = longSide))
            post { show(Shown(page, result)) }
        } catch (e: KilnException) {
            post {
                statsLine.text = "OCR failed: ${e.message}"
                setBusy(false)
            }
        }
    }

    /**
     * kiln does not apply a photo's EXIF orientation, and phone cameras store most photos sideways
     * with a "turn this" tag. Such a photo is turned here and handed over as a JPEG; any other
     * picture is handed over exactly as it came, so the numbers match `kiln-cli --ocr` on the file.
     */
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
        // kiln shrinks the page to 1600 pixels anyway, so 3200 keeps every detail it can use
        val src = decodeAtMost(bytes, bounds.outWidth, bounds.outHeight, 3200)
        val upright = Bitmap.createBitmap(src, 0, 0, src.width, src.height, turn, true)
        val jpeg = ByteArrayOutputStream().also { upright.compress(Bitmap.CompressFormat.JPEG, 95, it) }.toByteArray()
        val preview = if (max(upright.width, upright.height) > 2048) {
            val s = 2048f / max(upright.width, upright.height)
            Bitmap.createScaledBitmap(upright, (upright.width * s).toInt(), (upright.height * s).toInt(), true)
        } else upright
        return Page("$name (turned upright)", jpeg, preview, preview.width.toFloat() / upright.width)
    }

    private fun decodeAtMost(bytes: ByteArray, w: Int, h: Int, longest: Int): Bitmap {
        var sample = 1
        while (max(w, h) / (sample * 2) >= longest) sample *= 2
        val o = BitmapFactory.Options().apply { inSampleSize = sample }
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size, o)
    }

    // ---- showing the result ----------------------------------------------------------------

    private fun show(s: Shown) {
        shown = s
        val st = s.result.stats
        statsLine.text = buildString {
            append(String.format(Locale.US, "%d lines in %.0f ms", st.lines, st.ms))
            append(String.format(Locale.US, "  (finding %.0f · reading %.0f)", st.detectMs, st.readMs))
            append(String.format(Locale.US, "\n%s · %d × %d", s.page.name, st.width, st.height))
            if (st.linesFound != st.lines) append(" · ${st.linesFound - st.lines} boxes dropped as empty or junk")
            if (st.upsideDown) append(" · was upside down")
        }
        textOut.text = if (s.result.lines.isEmpty()) "(no text found)" else s.result.text
        drawPage(s)
        linesOut.removeAllViews()
        s.result.lines.forEachIndexed { n, line -> linesOut.addView(lineRow(n + 1, line)) }
        setBusy(false)
    }

    private fun drawPage(s: Shown) {
        val bmp = s.page.preview.copy(Bitmap.Config.ARGB_8888, true)
        if (showBoxes) {
            val c = Canvas(bmp)
            val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                style = Paint.Style.STROKE
                strokeWidth = max(2f, bmp.width / 400f)
            }
            val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
            val label = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                textSize = max(18f, bmp.width / 50f)
                typeface = Typeface.DEFAULT_BOLD
            }
            val k = s.page.scale
            s.result.lines.forEachIndexed { n, line ->
                val q = line.quad
                val path = Path().apply {
                    moveTo(q[0] * k, q[1] * k)
                    lineTo(q[2] * k, q[3] * k)
                    lineTo(q[4] * k, q[5] * k)
                    lineTo(q[6] * k, q[7] * k)
                    close()
                }
                val color = scriptColor(line.script)
                fill.color = (color and 0x00ffffff) or 0x30000000
                stroke.color = color
                label.color = color
                c.drawPath(path, fill)
                c.drawPath(path, stroke)
                c.drawText("${n + 1}", line.left * k, line.top * k - stroke.strokeWidth * 2, label)
            }
        }
        image.setImageBitmap(bmp)
    }

    private fun lineRow(n: Int, line: OcrLine): View = LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        setPadding(0, dp(4), 0, dp(4))
        addView(TextView(context).apply {
            text = String.format(Locale.US, "%2d  %.2f  %s", n, line.confidence, scriptName(line.script))
            typeface = Typeface.MONOSPACE
            textSize = 12f
            setTextColor(scriptColor(line.script))
            width = dp(150)
        })
        addView(TextView(context).apply {
            text = line.text
            textSize = 15f
            setTextIsSelectable(true)
            setTextColor(if (line.confidence < 0.6f) Color.rgb(0xb0, 0x30, 0x30) else Color.BLACK)
        }, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
    }

    private fun scriptColor(s: Script) = when (s) {
        Script.LATIN -> Color.rgb(0x1a, 0x73, 0xe8)
        Script.CJK -> Color.rgb(0xd9, 0x30, 0x25)
        Script.RTL -> Color.rgb(0x18, 0x80, 0x38)
        Script.OTHER -> Color.rgb(0x9c, 0x27, 0xb0)
    }

    private fun scriptName(s: Script) = when (s) {
        Script.LATIN -> "Latin"
        Script.CJK -> "CJK"
        Script.RTL -> "RTL"
        Script.OTHER -> "World"
    }

    // ---- where pictures come from ----------------------------------------------------------

    private val pickImage = 1
    private val takePhoto = 2
    private val pickModel = 3
    private val photoFile by lazy { PhotoProvider.file(this) }

    private fun openGallery() {
        val i = Intent(MediaStore.ACTION_PICK_IMAGES).setType("image/*")
        try {
            startActivityForResult(i, pickImage)
        } catch (_: Exception) {
            startActivityForResult(Intent(Intent.ACTION_GET_CONTENT).setType("image/*"), pickImage)
        }
    }

    private fun openCamera() {
        photoFile.delete()
        val uri = PhotoProvider.uri
        val i = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            .putExtra(MediaStore.EXTRA_OUTPUT, uri)
            .addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION or Intent.FLAG_GRANT_READ_URI_PERMISSION)
        try {
            startActivityForResult(i, takePhoto)
        } catch (_: Exception) {
            toast("No camera app found")
        }
    }

    private fun openSample() {
        val names = listOf("ocr_english.png", "ocr_bengali.png", "ocr_english_upside_down.png")
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
            pickModel -> data?.data?.let { importModel(it) }
        }
    }

    // ---- views -----------------------------------------------------------------------------

    private fun buildViews() {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(16), dp(16), dp(24))
        }
        root.addView(TextView(this).apply {
            text = "Lekha"
            textSize = 28f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(Color.BLACK)
        })
        root.addView(TextView(this).apply {
            text = "OCR on this phone with kiln and OCRia-nano-1. Nothing leaves the device."
            textSize = 13f
        })
        modelLine = TextView(this).apply {
            text = "Loading the model…"
            textSize = 12f
            setTextIsSelectable(true)
            setPadding(0, dp(8), 0, dp(8))
        }
        root.addView(modelLine)

        val camera = button("Camera") { openCamera() }
        val gallery = button("Gallery") { openGallery() }
        val sampleB = button("Sample") { openSample() }
        root.addView(row(camera, gallery, sampleB))

        val again = button("Read again") { lastPage?.let { p -> setBusy(true); statsLine.text = "Reading…"; worker.execute { runOcr(p) } } }
        longSideButton = button(longSideLabel()) {
            val steps = listOf(0, 1024, 2048, 800)
            longSide = steps[(steps.indexOf(longSide) + 1) % steps.size]
            longSideButton.text = longSideLabel()
        }
        boxesButton = button("Boxes: on") {
            showBoxes = !showBoxes
            boxesButton.text = if (showBoxes) "Boxes: on" else "Boxes: off"
            shown?.let { drawPage(it) }
        }
        root.addView(row(again, longSideButton, boxesButton))
        val modelB = button("Model file") {
            startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT).addCategory(Intent.CATEGORY_OPENABLE).setType("*/*"), pickModel)
        }
        val copy = button("Copy text") {
            val t = shown?.result?.text ?: return@button
            (getSystemService(CLIPBOARD_SERVICE) as ClipboardManager).setPrimaryClip(ClipData.newPlainText("Lekha", t))
            toast("Copied")
        }
        root.addView(row(modelB, copy))
        buttons = listOf(camera, gallery, sampleB, again, modelB)

        statsLine = TextView(this).apply {
            text = "Take a photo of printed text, pick one, or try a sample."
            textSize = 14f
            setTextColor(Color.BLACK)
            setPadding(0, dp(12), 0, dp(8))
        }
        root.addView(statsLine)
        image = ImageView(this).apply {
            adjustViewBounds = true
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(Color.rgb(0xee, 0xee, 0xee))
        }
        root.addView(image, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        root.addView(header("Text"))
        textOut = TextView(this).apply {
            textSize = 17f
            setTextColor(Color.BLACK)
            setTextIsSelectable(true)
        }
        root.addView(textOut)
        root.addView(header("Lines  (number, confidence, recognizer)"))
        linesOut = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        root.addView(linesOut)

        setContentView(ScrollView(this).apply {
            setBackgroundColor(Color.WHITE)
            fitsSystemWindows = true
            addView(root)
        })
        setBusy(true)
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

    private fun row(vararg views: View) = LinearLayout(this).apply {
        orientation = LinearLayout.HORIZONTAL
        gravity = Gravity.CENTER_VERTICAL
        views.forEach { addView(it, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)) }
    }

    private fun longSideLabel() = if (longSide == 0) "Size: 1600" else "Size: $longSide"

    private fun setBusy(busy: Boolean) {
        buttons.forEach { it.isEnabled = !busy }
        // the model button stays usable, so a missing model can be fixed while nothing is loaded
        buttons.last().isEnabled = true
    }

    private fun post(block: () -> Unit) = ui.post(block)
    private fun toast(t: String) = Toast.makeText(this, t, Toast.LENGTH_SHORT).show()
    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()
}
