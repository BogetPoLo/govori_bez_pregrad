package ru.obucheniesluzheniyu.app.ui

import android.content.ActivityNotFoundException
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.view.LayoutInflater
import android.view.View
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.media3.common.MediaItem
import androidx.media3.common.PlaybackException
import androidx.media3.common.Player
import androidx.media3.common.VideoSize
import androidx.media3.datasource.DefaultHttpDataSource
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.exoplayer.source.ProgressiveMediaSource
import androidx.media3.ui.PlayerView
import kotlinx.coroutines.delay
import org.json.JSONObject
import org.json.JSONTokener
import ru.obucheniesluzheniyu.app.R
import ru.obucheniesluzheniyu.app.phrases.SignEntry

@Composable
internal fun SignFlowVideoButton(entry: SignEntry) {
    var open by rememberSaveable(entry.signflowUrl) { mutableStateOf(false) }
    OutlinedButton(onClick = { open = true }) {
        Text(stringResource(R.string.signflow_watch))
    }
    if (open) {
        SignFlowVideoDialog(entry, onClose = { open = false })
    }
}

@Composable
private fun SignFlowVideoDialog(entry: SignEntry, onClose: () -> Unit) {
    val context = LocalContext.current
    var state by remember(entry.signflowUrl) { mutableStateOf<VideoState>(VideoState.Loading) }
    var webOnly by remember(entry.signflowUrl) { mutableStateOf(false) }
    var browserMissing by rememberSaveable { mutableStateOf(false) }

    Dialog(onDismissRequest = onClose) {
        BoxWithConstraints {
            val video = state as? VideoState.Ready
            val stream = state as? VideoState.Stream
            val ratio = video?.aspectRatio ?: stream?.aspectRatio
            val width = if (ratio == null) minOf(maxWidth, 320.dp) else {
                val maxVideoHeight = (maxHeight - 112.dp).coerceAtLeast(1.dp)
                minOf(maxWidth, 360.dp, maxVideoHeight * ratio)
            }
            Surface(modifier = Modifier.width(width), shape = MaterialTheme.shapes.large) {
                Column(Modifier.fillMaxWidth()) {
                    val status = when (state) {
                        VideoState.Loading, VideoState.WebLoading -> R.string.signflow_loading
                        VideoState.Unavailable -> R.string.signflow_load_error
                        is VideoState.Ready, is VideoState.Stream -> null
                    }
                    if (status != null) Text(stringResource(status), Modifier.padding(16.dp))
                    if (state != VideoState.Unavailable) {
                        AndroidView(
                            factory = { viewContext ->
                                WebView(viewContext).apply {
                                    // The public dictionary uses JavaScript for the player.
                                    settings.javaScriptEnabled = true
                                    settings.domStorageEnabled = true
                                    settings.mediaPlaybackRequiresUserGesture = false
                                    settings.allowFileAccess = false
                                    settings.allowContentAccess = false
                                    webViewClient = SignFlowPageClient(
                                        Uri.parse(entry.signflowUrl),
                                        webOnly = { webOnly },
                                        update = { state = it },
                                    )
                                    loadUrl(entry.signflowUrl)
                                }
                            },
                            update = { view ->
                                view.importantForAccessibility = if (video == null) {
                                    View.IMPORTANT_FOR_ACCESSIBILITY_NO_HIDE_DESCENDANTS
                                } else View.IMPORTANT_FOR_ACCESSIBILITY_AUTO
                            },
                            modifier = Modifier.fillMaxWidth()
                                .height(if (video == null) 1.dp else width / video.aspectRatio)
                                .alpha(if (video == null) 0f else 1f),
                            onRelease = {
                                (it.webViewClient as SignFlowPageClient).dispose(it)
                                it.stopLoading()
                                it.destroy()
                            },
                        )
                    }
                    if (stream != null && ratio != null) {
                        SignFlowStream(
                            source = stream.source,
                            lesson = entry.signflowUrl,
                            modifier = Modifier.fillMaxWidth().height(width / ratio),
                            onRatio = { actual ->
                                if ((state as? VideoState.Stream)?.source == stream.source) {
                                    state = stream.copy(aspectRatio = actual)
                                }
                            },
                            onError = {
                                webOnly = true
                                state = VideoState.WebLoading
                            },
                        )
                    }
                    if (browserMissing) Text(stringResource(R.string.signflow_no_browser), Modifier.padding(8.dp))
                    FlowRow(
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 4.dp),
                        horizontalArrangement = Arrangement.Center,
                    ) {
                        TextButton(onClick = {
                            try {
                                context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(entry.signflowUrl)))
                                onClose()
                            } catch (_: ActivityNotFoundException) {
                                browserMissing = true
                            }
                        }) {
                            Text(stringResource(R.string.signflow_open_browser))
                        }
                        TextButton(onClick = onClose) { Text(stringResource(R.string.signflow_close)) }
                    }
                }
            }
        }
    }
}

private sealed interface VideoState {
    data object Loading : VideoState
    data object WebLoading : VideoState
    data object Unavailable : VideoState
    data class Ready(val aspectRatio: Float) : VideoState
    data class Stream(val source: Uri, val aspectRatio: Float) : VideoState
}

@androidx.annotation.OptIn(androidx.media3.common.util.UnstableApi::class)
@Composable
private fun SignFlowStream(
    source: Uri,
    lesson: String,
    modifier: Modifier,
    onRatio: (Float) -> Unit,
    onError: () -> Unit,
) {
    val context = LocalContext.current
    val lifecycle = LocalLifecycleOwner.current.lifecycle
    val player = remember(source) { ExoPlayer.Builder(context).build() }
    var frameRendered by remember(source) { mutableStateOf(false) }

    DisposableEffect(player, lifecycle) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_STOP) player.pause()
        }
        val listener = object : Player.Listener {
            override fun onRenderedFirstFrame() {
                frameRendered = true
            }

            override fun onVideoSizeChanged(size: VideoSize) {
                if (size.width > 0 && size.height > 0) {
                    onRatio(size.width * size.pixelWidthHeightRatio / size.height)
                }
            }

            override fun onPlayerError(error: PlaybackException) = onError()
        }
        val http = DefaultHttpDataSource.Factory()
            .setDefaultRequestProperties(mapOf("Referer" to lesson))
        lifecycle.addObserver(observer)
        player.addListener(listener)
        player.setMediaSource(ProgressiveMediaSource.Factory(http).createMediaSource(MediaItem.fromUri(source)))
        player.playWhenReady = true
        player.prepare()
        onDispose {
            lifecycle.removeObserver(observer)
            player.removeListener(listener)
            player.release()
        }
    }
    LaunchedEffect(player, frameRendered) {
        if (!frameRendered) {
            delay(15_000)
            onError()
        }
    }
    if (!frameRendered) Text(stringResource(R.string.signflow_loading), Modifier.padding(8.dp))
    AndroidView(
        factory = { viewContext ->
            (LayoutInflater.from(viewContext).inflate(R.layout.sign_video_player, null) as PlayerView).apply {
                useController = true
                this.player = player
            }
        },
        modifier = if (frameRendered) modifier else Modifier.fillMaxWidth().height(1.dp).alpha(0f),
        onRelease = { it.player = null },
    )
}

private class SignFlowPageClient(
    private val lesson: Uri,
    private val webOnly: () -> Boolean,
    private val update: (VideoState) -> Unit,
) : WebViewClient() {
    private var failed = false
    private var disposed = false
    private var checkPlayer: Runnable? = null
    private var nativeSource: Uri? = null

    override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
        return request.isForMainFrame && !isLesson(request.url)
    }

    override fun onPageStarted(view: WebView, url: String, favicon: Bitmap?) {
        if (!failed) update(VideoState.Loading)
    }

    override fun onPageFinished(view: WebView, url: String) {
        if (failed || disposed) return
        if (!isLesson(Uri.parse(url))) {
            showError()
            return
        }
        checkPlayer?.let { view.removeCallbacks(it) }
        var secondsWithoutPlayer = 0
        var secondsBuffering = 0
        val poll = object : Runnable {
            override fun run() {
                if (nativeSource != null && !webOnly()) {
                    view.postDelayed(this, 1_000L)
                    return
                }
                view.evaluateJavascript("($PLAYER_STATUS)(${webOnly()})") { result ->
                    if (failed || disposed || checkPlayer !== this) return@evaluateJavascript
                    secondsWithoutPlayer = if (result == "null" || result == "\"missing\"") secondsWithoutPlayer + 1 else 0
                    secondsBuffering = if (result == "\"buffering\"") secondsBuffering + 1 else 0
                    if (result == "\"error\"" ||
                        secondsWithoutPlayer >= 15 || secondsBuffering >= 20
                    ) {
                        showError()
                        return@evaluateJavascript
                    }
                    if (result?.startsWith("\"{") == true) {
                        val payload = runCatching { JSONObject(JSONTokener(result).nextValue() as String) }.getOrNull()
                        if (payload != null) {
                            val ratio = payload.optDouble("ratio").toFloat().takeIf { it.isFinite() && it > 0f }
                            val source = Uri.parse(payload.optString("source"))
                            if (!webOnly() && source.scheme == "https" && !source.host.isNullOrBlank()) {
                                nativeSource = source
                                update(VideoState.Stream(source, ratio ?: 16f / 9f))
                            } else if (ratio != null) {
                                update(VideoState.Ready(ratio))
                            }
                        }
                    }
                    view.postDelayed(this, 1_000L)
                }
            }
        }
        checkPlayer = poll
        view.post(poll)
    }

    override fun onReceivedError(view: WebView, request: WebResourceRequest, error: WebResourceError) {
        if (request.isForMainFrame) showError()
    }

    override fun onReceivedHttpError(view: WebView, request: WebResourceRequest, response: WebResourceResponse) {
        if (request.isForMainFrame) showError()
    }

    private fun showError() {
        failed = true
        update(VideoState.Unavailable)
    }

    fun dispose(view: WebView) {
        disposed = true
        checkPlayer?.let { view.removeCallbacks(it) }
        checkPlayer = null
    }

    private fun isLesson(url: Uri): Boolean =
        url.scheme == lesson.scheme && url.authority == lesson.authority &&
            url.path?.trimEnd('/') == lesson.path

    private companion object {
        val PLAYER_STATUS = """
            (webOnly => {
                const videos = [...document.querySelectorAll('video')];
                if (!videos.length) return 'missing';
                const player = videos.find(video => video.hasAttribute('data-app-player') && !video.error)
                    || videos.find(video => !video.error && video.videoWidth && video.videoHeight
                        && video.getClientRects().length)
                    || videos.find(video => !video.error && video.videoWidth && video.videoHeight)
                    || videos.find(video => !video.error && video.currentSrc);
                if (!player) {
                    videos.filter(video => !video.error && video.preload === 'none')
                        .forEach(video => { video.preload = 'metadata'; });
                    return videos.every(video => video.error) ? 'error' : 'missing';
                }
                const source = player.currentSrc;
                const ratio = player.videoHeight ? player.videoWidth / player.videoHeight : null;
                if (!webOnly && source.startsWith('https://')
                    && !/\.(m3u8|mpd)(?:[?#]|$)/i.test(source)) {
                    return JSON.stringify({source, ratio});
                }

                if (!player.hasAttribute('data-app-player')) {
                    document.querySelectorAll('[data-app-player], [data-app-player-parent]').forEach(element => {
                        element.removeAttribute('data-app-player');
                        element.removeAttribute('data-app-player-parent');
                    });
                    player.setAttribute('data-app-player', '');
                    for (let parent = player.parentElement; parent; parent = parent.parentElement) {
                        parent.setAttribute('data-app-player-parent', '');
                    }
                }
                if (!document.getElementById('app-player-style')) {
                    const style = document.createElement('style');
                    style.id = 'app-player-style';
                    style.textContent = `
                        html, body { margin: 0 !important; overflow: hidden !important; background: black !important; }
                        body :not([data-app-player]):not([data-app-player-parent]) { display: none !important; }
                        body [data-app-player-parent] { display: contents !important; }
                        video[data-app-player] {
                            position: fixed !important; inset: 0 !important;
                            width: 100vw !important; height: 100vh !important;
                            max-width: none !important; max-height: none !important;
                            margin: 0 !important; padding: 0 !important; border: 0 !important;
                            border-radius: 0 !important; transform: none !important;
                            visibility: visible !important; opacity: 1 !important;
                            object-fit: contain !important; background: black !important;
                        }
                    `;
                    document.head.appendChild(style);
                }
                player.controls = true;
                player.playsInline = true;
                if (getComputedStyle(player).position !== 'fixed') return 'error';
                if (!player.hasAttribute('data-app-play-requested')) {
                    player.setAttribute('data-app-play-requested', '');
                    player.play().catch(() => {});
                }
                if (player.readyState < 2 || !ratio) return 'buffering';
                return JSON.stringify({ratio});
            })
        """.trimIndent()
    }
}
