package dev.kiln.receipt

import android.content.ContentProvider
import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import android.database.MatrixCursor
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.provider.OpenableColumns
import java.io.File

/**
 * The camera app runs as another app and cannot write into the app's folders, so the app hands it
 * a content:// address instead; the camera writes the photo through this provider into one file
 * in the cache. (androidx's FileProvider does the same, with more options than one file needs.)
 */
class PhotoProvider : ContentProvider() {
    companion object {
        val uri: Uri = Uri.parse("content://dev.kiln.receipt.photo/photo.jpg")
        fun file(ctx: Context) = File(ctx.cacheDir, "photos/photo.jpg").also { it.parentFile?.mkdirs() }
    }

    override fun onCreate() = true

    override fun openFile(uri: Uri, mode: String): ParcelFileDescriptor =
        ParcelFileDescriptor.open(file(context!!), ParcelFileDescriptor.parseMode(mode))

    override fun getType(uri: Uri) = "image/jpeg"

    override fun query(uri: Uri, projection: Array<String>?, selection: String?, args: Array<String>?, sort: String?): Cursor {
        val f = file(context!!)
        return MatrixCursor(arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE)).apply { addRow(arrayOf<Any>(f.name, f.length())) }
    }

    override fun insert(uri: Uri, values: ContentValues?): Uri? = null
    override fun update(uri: Uri, values: ContentValues?, selection: String?, args: Array<String>?) = 0
    override fun delete(uri: Uri, selection: String?, args: Array<String>?) = 0
}
