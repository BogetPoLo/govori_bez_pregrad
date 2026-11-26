package ru.obucheniesluzheniyu.app.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LifecycleResumeEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import ru.obucheniesluzheniyu.app.R
import ru.obucheniesluzheniyu.app.recognition.GestureCamera
import ru.obucheniesluzheniyu.app.recognition.RecognitionState
import ru.obucheniesluzheniyu.app.recognition.RecognitionViewModel

@Composable
internal fun RecognitionScreen(model: RecognitionViewModel = viewModel()) {
    val state by model.state.collectAsStateWithLifecycle()
    Card(Modifier.fillMaxWidth().padding(24.dp)) {
        Column(Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            when (val current = state) {
                RecognitionState.Loading -> Text(stringResource(R.string.recognition_loading))
                RecognitionState.Downloading -> Text(stringResource(R.string.recognition_downloading))
                RecognitionState.Unavailable -> {
                    Text(
                        stringResource(R.string.recognition_unavailable),
                        style = MaterialTheme.typography.titleLarge,
                    )
                    Text(stringResource(R.string.recognition_unavailable_detail))
                    Button(onClick = model::retry) { Text(stringResource(R.string.recognition_retry)) }
                }
                RecognitionState.Processing -> Text(stringResource(R.string.recognition_processing))
                RecognitionState.Retry -> {
                    Text(stringResource(R.string.recognition_try_again))
                    Button(onClick = model::retry) { Text(stringResource(R.string.recognition_retry)) }
                }
                is RecognitionState.Result -> {
                    Text(current.entry.text, style = MaterialTheme.typography.headlineSmall)
                    SignFlowVideoButton(current.entry)
                    OutlinedButton(onClick = model::retry) { Text(stringResource(R.string.recognition_retry)) }
                }
                RecognitionState.Ready, RecognitionState.Recording -> {
                    GestureCaptureControls(model, current == RecognitionState.Recording)
                }
            }
        }
    }
}

@Composable
private fun GestureCaptureControls(model: RecognitionViewModel, recording: Boolean) {
    val context = LocalContext.current
    val permission = Manifest.permission.CAMERA
    var allowed by remember { mutableStateOf(false) }
    var denied by remember { mutableStateOf(false) }
    val requestPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {
        allowed = it
        denied = !it
    }
    LifecycleResumeEffect(context) {
        allowed = ContextCompat.checkSelfPermission(context, permission) == PackageManager.PERMISSION_GRANTED
        onPauseOrDispose { model.cancelCapture() }
    }

    if (!allowed) {
        Text(stringResource(if (denied) R.string.camera_denied else R.string.camera_permission))
        Button(onClick = { requestPermission.launch(permission) }) {
            Text(stringResource(R.string.camera_allow))
        }
        if (denied) {
            OutlinedButton(onClick = {
                context.startActivity(
                    Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.fromParts("package", context.packageName, null)),
                )
            }) { Text(stringResource(R.string.camera_settings)) }
        }
        return
    }

    var streaming by remember { mutableStateOf(false) }
    var failed by remember { mutableStateOf(false) }
    if (failed) {
        Text(stringResource(R.string.camera_unavailable))
        Button(onClick = { failed = false }) { Text(stringResource(R.string.recognition_retry)) }
        return
    }

    GestureCamera(
        model = model,
        onStreaming = { streaming = it },
        onError = {
            streaming = false
            failed = true
        },
        modifier = Modifier.fillMaxWidth().aspectRatio(4f / 3f),
    )
    Text(stringResource(if (recording) R.string.recognition_recording else R.string.recognition_instructions))
    Button(
        onClick = if (recording) model::finishCapture else model::startCapture,
        enabled = streaming,
    ) {
        Text(stringResource(if (recording) R.string.recognition_stop else R.string.recognition_start))
    }
}
