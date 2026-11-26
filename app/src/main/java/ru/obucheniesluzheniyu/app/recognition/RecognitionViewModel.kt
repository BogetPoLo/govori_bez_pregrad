package ru.obucheniesluzheniyu.app.recognition

import android.app.Application
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import ru.obucheniesluzheniyu.app.phrases.SignEntry
import ru.obucheniesluzheniyu.app.phrases.loadSignEntries

internal sealed interface RecognitionState {
    data object Loading : RecognitionState
    data object Downloading : RecognitionState
    data object Unavailable : RecognitionState
    data object Ready : RecognitionState
    data object Recording : RecognitionState
    data object Processing : RecognitionState
    data object Retry : RecognitionState
    data class Result(val entry: SignEntry) : RecognitionState
}

internal class RecognitionViewModel(application: Application) : AndroidViewModel(application) {
    private val capture = GestureCapture()
    private val attempts = Channel<Attempt>()

    @Volatile
    private var recipe: FrameRecipe? = null
    private val mutableState = MutableStateFlow<RecognitionState>(RecognitionState.Loading)
    val state = mutableState.asStateFlow()
    private var loadingJob: Job? = null

    init {
        loadModel()
    }

    private fun loadModel() {
        if (loadingJob?.isActive == true) return
        mutableState.value = RecognitionState.Loading
        loadingJob = viewModelScope.launch {
            var model: SlovoModel? = null
            try {
                withContext(Dispatchers.IO) {
                    val application = getApplication<Application>()
                    model = SlovoModel.open(application, loadSignEntries(application.assets)) {
                        mutableState.value = RecognitionState.Downloading
                    }
                }
                val loaded = checkNotNull(model)
                recipe = loaded.recipe
                mutableState.value = RecognitionState.Ready
                for (attempt in attempts) {
                    try {
                        val entry = withContext(Dispatchers.Default) { loaded.recognize(attempt.frames) }
                        mutableState.value = RecognitionState.Result(entry)
                        Log.i(TAG, "Capture end to candidate: ${SystemClock.elapsedRealtime() - attempt.endedAtMs} ms")
                    } catch (error: Exception) {
                        if (error is CancellationException) throw error
                        Log.w(TAG, "Gesture inference failed", error)
                        mutableState.value = RecognitionState.Retry
                    }
                }
            } catch (error: Exception) {
                if (error is CancellationException) throw error
                Log.w(TAG, "Slovo MViT unavailable", error)
                mutableState.value = RecognitionState.Unavailable
            } catch (error: LinkageError) {
                Log.w(TAG, "ONNX Runtime unavailable", error)
                mutableState.value = RecognitionState.Unavailable
            } finally {
                capture.cancel()
                recipe = null
                withContext(NonCancellable + Dispatchers.IO) { model?.close() }
            }
        }
    }

    fun startCapture() {
        if (recipe == null) return
        if (mutableState.value != RecognitionState.Ready) return
        capture.start()
        mutableState.value = RecognitionState.Recording
    }

    fun finishCapture() {
        if (mutableState.value != RecognitionState.Recording) return
        val stoppedAtNs = SystemClock.elapsedRealtimeNanos()
        val frames = capture.finish(stoppedAtNs)
        if (frames.isEmpty()) {
            mutableState.value = RecognitionState.Retry
            return
        }
        mutableState.value = RecognitionState.Processing
        viewModelScope.launch {
            attempts.send(Attempt(frames, stoppedAtNs / 1_000_000L))
        }
    }

    fun analyze(image: ImageProxy) {
        try {
            recipe?.let { capture.analyze(image, it) }
        } catch (error: Exception) {
            Log.w(TAG, "Camera frame could not be processed", error)
        } finally {
            image.close()
        }
    }

    fun cancelCapture() {
        if (mutableState.value != RecognitionState.Recording) return
        capture.cancel()
        mutableState.value = RecognitionState.Ready
    }

    fun retry() {
        when (mutableState.value) {
            RecognitionState.Unavailable -> loadModel()
            RecognitionState.Retry, is RecognitionState.Result -> {
                mutableState.value = RecognitionState.Ready
            }
            else -> Unit
        }
    }

    override fun onCleared() {
        attempts.cancel()
        super.onCleared()
    }

    private data class Attempt(val frames: List<GestureFrame>, val endedAtMs: Long)

    private companion object {
        const val TAG = "SignFlow"
    }
}
