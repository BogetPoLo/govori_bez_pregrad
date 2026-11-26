package ru.obucheniesluzheniyu.app.ui

import android.net.Uri
import android.view.LayoutInflater
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.key
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import ru.obucheniesluzheniyu.app.R
import ru.obucheniesluzheniyu.app.phrases.PhrasesViewModel

@Composable
internal fun PhrasesScreen(model: PhrasesViewModel = viewModel()) {
    val query by model.query.collectAsStateWithLifecycle()
    val selected = model.selected(query)
    val matches = model.search(query)
    val focusManager = LocalFocusManager.current
    Column(
        modifier = Modifier.fillMaxWidth().padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        OutlinedTextField(
            value = query,
            onValueChange = model::updateQuery,
            modifier = Modifier.fillMaxWidth(),
            label = { Text(stringResource(R.string.phrases_search)) },
            trailingIcon = if (query.isNotEmpty()) {
                { TextButton(onClick = { model.updateQuery("") }) { Text(stringResource(R.string.phrases_clear)) } }
            } else {
                null
            },
            singleLine = true,
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
        )

        val videoAsset = selected?.videoAsset
        when {
            videoAsset != null -> key(videoAsset) { LocalSignVideo(videoAsset) }
            selected != null -> SignFlowVideoButton(selected)
            matches.isEmpty() -> Text(stringResource(R.string.phrases_video_unavailable))
            else -> matches.forEach { entry ->
                OutlinedButton(
                    onClick = {
                        model.updateQuery(entry.text)
                        focusManager.clearFocus()
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(entry.text)
                }
            }
        }
    }
}

@androidx.annotation.OptIn(androidx.media3.common.util.UnstableApi::class)
@Composable
private fun LocalSignVideo(asset: String) {
    val context = LocalContext.current
    val lifecycle = LocalLifecycleOwner.current.lifecycle
    val player = remember(context) { ExoPlayer.Builder(context).build() }
    var failed by remember { mutableStateOf(false) }

    DisposableEffect(player, lifecycle) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_STOP) player.pause()
        }
        val listener = object : Player.Listener {
            override fun onPlayerError(error: PlaybackException) {
                failed = true
            }
        }
        lifecycle.addObserver(observer)
        player.addListener(listener)
        player.setMediaItem(MediaItem.fromUri(Uri.Builder().scheme("asset").path("/$asset").build()))
        player.prepare()
        player.play()

        onDispose {
            lifecycle.removeObserver(observer)
            player.removeListener(listener)
            player.release()
        }
    }

    if (failed) {
        Text(stringResource(R.string.phrases_video_unavailable))
    } else {
        AndroidView(
            factory = {
                (LayoutInflater.from(it).inflate(R.layout.sign_video_player, null) as PlayerView).apply {
                    this.player = player
                }
            },
            modifier = Modifier.fillMaxWidth().aspectRatio(1f),
            onRelease = { it.player = null },
        )
        Text(stringResource(R.string.phrases_video_preliminary))
        OutlinedButton(onClick = {
            player.seekTo(0)
            player.play()
        }) {
            Text(stringResource(R.string.phrases_repeat))
        }
    }
}
