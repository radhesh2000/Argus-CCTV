/**
 * Talking to go2rtc — through the appliance's own front door.
 *
 * This file used to open a connection straight to go2rtc on `:1984`, which is
 * why go2rtc had to be told `api.origin: "*"`, which is why a second listener
 * with no credentials on it served a snapshot of every camera to anything that
 * could route to the plant LAN. Phase 5 put go2rtc's HTTP API on loopback and
 * moved both calls here onto `/api/gateway/*`: same origin, same session
 * cookie, same role check as every other route.
 *
 * The change is one seam wide — `RELAY` below — because everything downstream
 * is go2rtc's own signalling protocol, relayed frame-for-frame by the API and
 * therefore unchanged.
 *
 * **The media does not come through here.** ICE connects the browser directly
 * to go2rtc's `webrtc_port` on this host, and relaying that would mean running
 * a TURN server on the appliance. See docs/SECURITY.md; the payload is
 * per-session SRTP whose keys were negotiated over this authenticated channel.
 */

/** The seam. Same origin in production and in dev — vite already proxies
 *  `/api`, so the signalling WebSocket needs no proxy rule of its own any
 *  more, and there is no port for a deployment to get wrong. */
export function gatewayBase(): string {
  return '/api/gateway';
}

function websocketUrl(stream: string): string {
  const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${scheme}//${window.location.host}${gatewayBase()}/ws?src=${encodeURIComponent(stream)}`;
}

/** A single still from the stream — used as the poster while WebRTC negotiates,
 *  so a wall of twelve tiles shows the plant immediately rather than twelve
 *  black rectangles. The browser sends the session cookie with an `<img>` load
 *  the same as with a fetch, so this needs nothing but the URL. */
export function frameUrl(stream: string): string {
  return `${gatewayBase()}/frame.jpeg?src=${encodeURIComponent(stream)}`;
}

export type PlayerState = 'connecting' | 'playing' | 'failed';

/**
 * go2rtc's own signalling handshake, as implemented by its bundled
 * `video-rtc.js` (verified against go2rtc 1.9.14 on this appliance):
 *
 *   -> {"type":"webrtc/offer","value":"<sdp>"}
 *   <- {"type":"webrtc/answer","value":"<sdp>"}
 *   <> {"type":"webrtc/candidate","value":"<candidate>"}   (both directions)
 *
 * Trickle ICE, hence the WebSocket rather than the single-shot POST endpoint:
 * on a plant LAN the first host candidate is usually the one that works, and
 * waiting for gathering to complete adds a visible second to every tile. It is
 * also why the API relays this socket rather than reimplementing the exchange
 * — the protocol stays go2rtc's.
 */
export function connectWebRtc(
  stream: string,
  video: HTMLVideoElement,
  onState: (state: PlayerState) => void,
): () => void {
  let closed = false;
  const pc = new RTCPeerConnection({
    iceServers: [],
    // Everything is on the LAN; bundling keeps it to one transport.
    bundlePolicy: 'max-bundle',
  });
  const socket = new WebSocket(websocketUrl(stream));

  const send = (message: unknown) => {
    if (socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
  };

  pc.addEventListener('icecandidate', (event) => {
    send({ type: 'webrtc/candidate', value: event.candidate ? event.candidate.candidate : '' });
  });

  pc.addEventListener('track', (event) => {
    if (event.streams[0]) video.srcObject = event.streams[0];
  });

  pc.addEventListener('connectionstatechange', () => {
    if (closed) return;
    if (pc.connectionState === 'connected') onState('playing');
    else if (pc.connectionState === 'failed' || pc.connectionState === 'disconnected') {
      onState('failed');
    }
  });

  socket.addEventListener('open', () => {
    // recvonly: this is a viewer, and offering to send would make the gateway
    // negotiate an upstream it will never receive.
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.createOffer()
      .then((offer) => pc.setLocalDescription(offer).then(() => offer))
      .then((offer) => send({ type: 'webrtc/offer', value: offer.sdp }))
      .catch(() => onState('failed'));
  });

  socket.addEventListener('message', (event) => {
    let message: { type?: string; value?: string };
    try {
      message = JSON.parse(String(event.data)) as { type?: string; value?: string };
    } catch {
      return;
    }
    if (message.type === 'webrtc/answer' && message.value) {
      pc.setRemoteDescription({ type: 'answer', sdp: message.value }).catch(() => onState('failed'));
    } else if (message.type === 'webrtc/candidate' && message.value) {
      pc.addIceCandidate({ candidate: message.value, sdpMid: '0' }).catch(() => {
        /* a candidate we cannot use is not a failure while others remain */
      });
    } else if (message.type === 'error') {
      onState('failed');
    }
  });

  socket.addEventListener('error', () => onState('failed'));
  socket.addEventListener('close', () => {
    if (!closed && pc.connectionState !== 'connected') onState('failed');
  });

  return () => {
    closed = true;
    try {
      socket.close();
    } catch {
      /* already gone */
    }
    pc.getSenders().forEach((sender) => sender.track?.stop());
    pc.close();
    video.srcObject = null;
  };
}
