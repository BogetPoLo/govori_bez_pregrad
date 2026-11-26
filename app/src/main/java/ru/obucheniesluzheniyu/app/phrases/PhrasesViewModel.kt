package ru.obucheniesluzheniyu.app.phrases

import android.app.Application
import android.content.res.AssetManager
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.SavedStateHandle
import org.json.JSONArray
import java.util.Locale

internal data class SignEntry(val text: String, val signflowUrl: String, val videoAsset: String?, val classIndex: Int)

internal class PhrasesViewModel(application: Application, private val state: SavedStateHandle) :
    AndroidViewModel(application) {
    val query = state.getStateFlow("phraseQuery", "")
    private val entries = loadSignEntries(application.assets)

    fun search(text: String) = entries.filter { it.text.normalized().contains(text.normalized()) }

    fun selected(text: String) = entries.firstOrNull { it.text.normalized() == text.normalized() }

    fun updateQuery(text: String) {
        state["phraseQuery"] = text
    }

    private fun String.normalized() = trim().lowercase(Locale.ROOT).replace('ё', 'е').replace(whitespace, " ")

    private companion object {
        val whitespace = Regex("\\s+")
    }
}

internal fun loadSignEntries(assets: AssetManager): List<SignEntry> {
    val videos = assets.open("phrases/catalog.json").bufferedReader().use { JSONArray(it.readText()) }
    val availableFiles = assets.list("phrases/avatars").orEmpty().toSet()
    return List(videos.length()) { index ->
        val entry = videos.getJSONObject(index)
        val file = entry.optString("file")
        SignEntry(
            text = entry.getString("word"),
            signflowUrl = "https://signflow.ru/${entry.getString("signflow")}",
            videoAsset = file.takeIf { it in availableFiles }?.let { "phrases/avatars/$it" },
            classIndex = entry.getInt("classIndex"),
        )
    }
}
