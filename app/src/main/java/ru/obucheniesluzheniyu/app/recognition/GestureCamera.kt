package ru.obucheniesluzheniyu.app.recognition

import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.Observer
import androidx.lifecycle.compose.LocalLifecycleOwner
import java.util.concurrent.Executors

@Composable
internal fun GestureCamera(
    model: RecognitionViewModel,
    onStreaming: (Boolean) -> Unit,
    onError: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val view = remember(context) {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }

    AndroidView(factory = { view }, modifier = modifier)
    DisposableEffect(view, owner, model) {
        val mainExecutor = ContextCompat.getMainExecutor(context)
        val analyzer = Executors.newSingleThreadExecutor()
        val providerFuture = ProcessCameraProvider.getInstance(context)
        val preview = Preview.Builder().build().also { it.surfaceProvider = view.surfaceProvider }
        val analysis = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()
        val streamObserver = Observer<PreviewView.StreamState> {
            val streaming = it == PreviewView.StreamState.STREAMING
            onStreaming(streaming)
            if (!streaming) model.cancelCapture()
        }
        val lifecycleObserver = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_STOP) model.cancelCapture()
        }
        view.previewStreamState.observe(owner, streamObserver)
        owner.lifecycle.addObserver(lifecycleObserver)
        var disposed = false
        var provider: ProcessCameraProvider? = null
        providerFuture.addListener({
            if (!disposed) {
                try {
                    val loaded = providerFuture.get()
                    provider = loaded
                    val selector = if (loaded.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA)) {
                        CameraSelector.DEFAULT_FRONT_CAMERA
                    } else {
                        CameraSelector.DEFAULT_BACK_CAMERA
                    }
                    val rotation = view.display?.rotation ?: android.view.Surface.ROTATION_0
                    preview.targetRotation = rotation
                    analysis.targetRotation = rotation
                    analysis.setAnalyzer(analyzer, model::analyze)
                    loaded.bindToLifecycle(owner, selector, preview, analysis)
                } catch (error: Exception) {
                    Log.w("SignFlow", "Camera binding failed", error)
                    onError()
                }
            }
        }, mainExecutor)

        onDispose {
            disposed = true
            model.cancelCapture()
            view.previewStreamState.removeObserver(streamObserver)
            owner.lifecycle.removeObserver(lifecycleObserver)
            analysis.clearAnalyzer()
            provider?.unbind(preview, analysis)
            analyzer.shutdown()
        }
    }
}
