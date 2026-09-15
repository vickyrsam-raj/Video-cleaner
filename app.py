import os, re, time, glob, subprocess, threading, uuid
import numpy as np, cv2
from flask import Flask, request, jsonify, send_file, Response

FF = None
def ffmpeg_bin():
    global FF
    if FF is None:
        import imageio_ffmpeg
        FF = imageio_ffmpeg.get_ffmpeg_exe()
    return FF

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
JOBS = {}

RANGES = {
  'yellow': [((22, 180, 180), (38, 255, 255))],
  'red':    [((0, 190, 150), (8, 255, 255)), ((172, 190, 150), (179, 255, 255))],
  'white':  [((0, 0, 220), (179, 80, 255))],
  'orange': [((10, 150, 150), (22, 255, 255))],
  'cyan':   [((85, 120, 150), (130, 255, 255))],
  'green':  [((40, 120, 120), (85, 255, 255))],
}
MIN_AREA, MAX_AREA = 8, 12000

def text_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = np.zeros(img.shape[:2], np.uint8)
    for cname, rs in RANGES.items():
        raw = np.zeros_like(m)
        for r in rs:
            lo, hi = r[0], r[1]
            maxa = r[2] if len(r) > 2 else MAX_AREA
            raw = cv2.inRange(hsv, np.array(lo), np.array(hi))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(raw)
            for k in range(1, n):
                a = int(stats[k, cv2.CC_STAT_AREA])
                if MIN_AREA <= a <= maxa:
                    comp = (lab == k).astype(np.uint8)
                    ring = cv2.dilate(comp, np.ones((7, 7), np.uint8)) - comp
                    rsz = int(ring.sum())
                    if rsz == 0 or (hsv[..., 2] < 120)[ring > 0].mean() >= 0.5:
                        m[lab == k] = 255
    # left-column rule: dim white list numbers (top 62% of left 13%)
    raw = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([179, 95, 255]))
    raw[int(img.shape[0] * 0.66):, :] = 0
    raw[:, int(img.shape[1] * 0.30):] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(raw)
    for k in range(1, n):
        a = int(stats[k, cv2.CC_STAT_AREA])
        if 5 <= a <= 900:
            comp = (lab == k).astype(np.uint8)
            ring = cv2.dilate(comp, np.ones((7, 7), np.uint8)) - comp
            rsz = int(ring.sum())
            if rsz and (hsv[..., 2] < 120)[ring > 0].mean() >= 0.35:
                m[lab == k] = 255
    # bright small handle on dark bg, bottom center
    raw = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([179, 90, 255]))
    raw[:int(img.shape[0] * 0.80), :] = 0
    raw[:, :int(img.shape[1] * 0.25)] = 0
    raw[:, int(img.shape[1] * 0.75):] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(raw)
    for k in range(1, n):
        a = int(stats[k, cv2.CC_STAT_AREA])
        hh = int(stats[k, cv2.CC_STAT_HEIGHT])
        if 5 <= a <= 500 and hh <= 22:
            comp = (lab == k).astype(np.uint8)
            ring = cv2.dilate(comp, np.ones((7, 7), np.uint8)) - comp
            if ring.sum() and hsv[..., 2][lab == k].mean() >= hsv[..., 2][ring > 0].mean() + 40:
                m[lab == k] = 255
    # bottom-strip rule: white date stamps / handles (no dark outline needed)
    h = img.shape[0]
    raw = cv2.inRange(hsv, np.array([0, 0, 95]), np.array([179, 80, 205]))
    raw[:int(h * 0.80), :] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(raw)
    for k in range(1, n):
        a = int(stats[k, cv2.CC_STAT_AREA])
        hh = int(stats[k, cv2.CC_STAT_HEIGHT])
        if 5 <= a <= 3000 and hh <= 60:
            comp = (lab == k).astype(np.uint8)
            ring = cv2.dilate(comp, np.ones((7, 7), np.uint8)) - comp
            if ring.sum() and hsv[..., 2][lab == k].mean() + 25 <= hsv[..., 2][ring > 0].mean():
                m[lab == k] = 255
    k = 5 if img.shape[1] < 480 else 9
    return cv2.dilate(m, np.ones((k, k), np.uint8))

def probe(path):
    r = subprocess.run([ffmpeg_bin(), '-i', path], capture_output=True, text=True)
    err = r.stderr
    m = re.search(r', (\d{2,5})x(\d{2,5})', err)
    W, H = int(m.group(1)), int(m.group(2))
    fm = re.search(r'([\d.]+) fps', err)
    fps = float(fm.group(1)) if fm else 30.0
    return W, H, fps

def frames_iter(path, W, H):
    p = subprocess.Popen([ffmpeg_bin(), '-v', 'error', '-i', path, '-f', 'rawvideo',
                          '-pix_fmt', 'bgr24', '-'],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    sz = W * H * 3
    while True:
        data = p.stdout.read(sz)
        if len(data) != sz:
            break
        yield np.frombuffer(data, np.uint8).reshape(H, W, 3)
    p.stdout.close(); p.wait()

def process(job):
    j = JOBS[job]
    try:
        d = j['dir']; inp = d + '/in.mp4'
        W, H, fps = probe(inp)
        j.update(W=W, H=H)
        # PASS 1: masks
        counts = []
        i = 0
        for fr in frames_iter(inp, W, H):
            m = text_mask(fr)
            cv2.imwrite(f"{d}/masks/{i:06d}.png", m)
            counts.append(int(m.sum() // 255)); i += 1
            j['progress'] = int(30 * i / max(1, i + 1)); j['stage'] = f'Scan {i}'
        N = len(counts); counts = np.array(counts)
        j['total'] = N
        clean = np.where(counts == 0)[0]
        j['textframes'] = int((counts > 0).sum())
        # PASS 2: plate
        use = clean[::max(1, len(clean) // 120)][:120] if len(clean) >= 5 else np.argsort(counts)[:60]
        want = set(int(x) for x in use)
        stack = []
        for idx, fr in enumerate(frames_iter(inp, W, H)):
            if idx in want:
                stack.append(cv2.resize(fr, (W // 2, H // 2)))
            if idx > max(want):
                break
        plate = cv2.resize(np.median(np.stack(stack), axis=0).astype(np.uint8), (W, H))
        plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY).astype(int)
        j['stage'] = 'Rebuilding background'
        # PASS 3: composite + encode
        enc = subprocess.Popen([ffmpeg_bin(), '-y', '-v', 'error',
                                '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
                                '-r', str(fps), '-i', 'pipe:0', '-i', inp,
                                '-map', '0:v:0', '-map', '1:a?',
                                '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '20',
                                '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest',
                                d + '/out.mp4'], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for i, fr in enumerate(frames_iter(inp, W, H)):
            m = cv2.imread(f"{d}/masks/{i:06d}.png", cv2.IMREAD_GRAYSCALE)
            if m is not None and m.sum() > 0:
                gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(int)
                fg = np.abs(gray - plate_gray) > 45                      # moving stuff (people)
                fg_d = cv2.dilate(fg.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                filled = fr.copy(); filled[m > 0] = plate[m > 0]        # real background plate
                if (fg_d & (m > 0)).sum() > 0.1 * (m > 0).sum():        # person behind text?
                    telea = cv2.inpaint(fr, m, 8, cv2.INPAINT_TELEA)
                    sel = fg_d & (m > 0)
                    filled[sel] = telea[sel]                            # keep person continuous
                mf = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (7, 7), 0)[..., None]
                fr = (fr * (1 - mf) + filled * mf).astype(np.uint8)
            enc.stdin.write(fr.tobytes())
            j['progress'] = 30 + int(65 * (i + 1) / N); j['stage'] = f'Cleaning {i+1}/{N}'
        enc.stdin.close(); enc.wait()
        j['status'] = 'done'; j['progress'] = 100; j['stage'] = 'Done!'
    except Exception as e:
        j['status'] = 'error'; j['stage'] = f'Error: {e}'

@app.route('/')
def index():
    items = sorted(glob.glob('/home/user/CLEAN_*.mp4'))
    if items:
        lst = ''.join(
            f'<a class=dl href="/files/{os.path.basename(p)}">⬇ {os.path.basename(p)}</a><br>'
            for p in items)
    else:
        lst = '<p style="color:#94a3b6">No clean videos yet.</p>'
    return Response(INDEX.replace('<!--LIST-->', lst), mimetype='text/html')

@app.route('/files/<name>')
def files(name):
    if not re.match(r'^CLEAN_[A-Za-z0-9_]+\.mp4$', name):
        return jsonify({'error': 'not found'}), 404
    p = '/home/user/' + name
    if not os.path.exists(p):
        return jsonify({'error': 'not found'}), 404
    return send_file(p, as_attachment=True, download_name=name)

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('video')
    if not f:
        return jsonify({'error': 'no video'}), 400
    job = uuid.uuid4().hex[:8]
    d = f'/tmp/job_{job}'; os.makedirs(d + '/masks', exist_ok=True)
    f.save(d + '/in.mp4')
    JOBS[job] = {'status': 'running', 'progress': 0, 'dir': d, 'stage': 'Starting'}
    threading.Thread(target=process, args=(job,), daemon=True).start()
    return jsonify({'job': job})

@app.route('/status/<job>')
def status(job):
    j = JOBS.get(job)
    return jsonify(j if j else {'status': 'unknown'})

@app.route('/download/<job>')
def download(job):
    j = JOBS.get(job)
    if not j or j['status'] != 'done':
        return jsonify({'error': 'not ready'}), 400
    return send_file(j['dir'] + '/out.mp4', as_attachment=True, download_name='CLEAN_VIDEO.mp4')

INDEX = """<!doctype html><html><head><meta name=viewport content="width=device-width,initial-scale=1">
<title>Video Caption, Watermark, Text & Logo Remover</title><style>
body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:24px;display:flex;flex-direction:column;align-items:center}
h1{font-size:22px} .card{background:#1e293b;border-radius:16px;padding:24px;width:min(480px,92vw);text-align:center}
input[type=file]{margin:12px 0} button{background:#22c55e;color:#052e16;border:0;border-radius:10px;padding:12px 22px;font-size:16px;font-weight:700}
button:disabled{background:#475569;color:#94a3b8}
.bar{height:12px;background:#334155;border-radius:6px;overflow:hidden;margin:14px 0}
.fill{height:100%;width:0%;background:#22c55e;transition:width .3s}
#msg{min-height:22px;color:#94a3b8} a.dl{display:inline-block;margin-top:10px;background:#3b82f6;color:#fff;padding:12px 22px;border-radius:10px;text-decoration:none;font-weight:700}
</style></head><body>
<!-- v2 --><h1>🎬 Video Caption, Watermark, Text & Logo Remover</h1>
<div class=card>
<p>Removes burned-in captions, subtitles, title bars, word cards, numbered lists, watermarks, @handles and corner logos. Background rebuilt for real (no blur), audio kept. Works on phone — up to 10 min.</p>
<input type=file id=f accept=video/*><br>
<button id=go onclick=up()>Remove captions</button>
<div class=bar><div class=fill id=fill></div></div>
<div id=msg>Pick a video (up to 10 min).</div>
<div id=dl></div>
</div>
<div class=card style="margin-top:16px">
<h2 style="font-size:18px;margin:0 0 10px">📁 Your clean videos (tap to download)</h2>
<!--LIST-->
</div>
<script>
async function up(){
 const fd=new FormData(); fd.append('video',f.files[0]);
 go.disabled=true; msg.textContent='Uploading...';
 const r=await fetch('/upload',{method:'POST',body:fd}); const j=await r.json();
 poll(j.job);
}
function poll(job){
 const t=setInterval(async()=>{
  const r=await fetch('/status/'+job); const j=await r.json();
  fill.style.width=j.progress+'%'; msg.textContent=j.stage||j.status;
  if(j.status=='done'){clearInterval(t);go.disabled=false;
    dl.innerHTML='<a class=dl href="/download/'+job+'">⬇ Download CLEAN_VIDEO.mp4</a>';}
  if(j.status=='error'){clearInterval(t);go.disabled=false;}
 },1000);
}
</script></body></html>"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), threaded=True)
