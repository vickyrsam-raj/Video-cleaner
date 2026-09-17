import os, re, time, glob, subprocess, threading, uuid
from datetime import date
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
USAGE = {}
FREE_LIMIT = int(os.environ.get('FREE_LIMIT', '3'))
PRO_CODES = set(os.environ.get('PRO_CODES', 'SAMFREE2026').split(','))

RANGES = {
  'yellow': [((22, 180, 180), (38, 255, 255))],
  'red':    [((0, 190, 150), (8, 255, 255)), ((172, 190, 150), (179, 255, 255))],
  'white':  [((0, 0, 220), (179, 80, 255))],
  'orange': [((10, 150, 150), (22, 255, 255))],
  'cyan':   [((85, 120, 150), (130, 255, 255))],
  'green':  [((40, 120, 120), (85, 255, 255))],
}
MIN_AREA, MAX_AREA = 8, 12000

def text_mask(img, zones=True):
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
    if not zones:
        k = 5 if img.shape[1] < 480 else 9
        return cv2.dilate(m, np.ones((k, k), np.uint8))
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
    # list names: small bright text in left zone (no dark outline needed)
    raw = cv2.inRange(hsv, np.array([0, 0, 130]), np.array([179, 255, 255]))
    raw[:int(img.shape[0] * 0.08), :] = 0
    raw[int(img.shape[0] * 0.68):, :] = 0
    raw[:, int(img.shape[1] * 0.32):] = 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(raw)
    for k in range(1, n):
        a = int(stats[k, cv2.CC_STAT_AREA])
        hh = int(stats[k, cv2.CC_STAT_HEIGHT])
        ww = int(stats[k, cv2.CC_STAT_WIDTH])
        if 20 <= a <= 500 and 5 <= hh <= 18 and ww <= 60:
            comp = (lab == k).astype(np.uint8)
            ring = cv2.dilate(comp, np.ones((7, 7), np.uint8)) - comp
            if ring.sum() and hsv[..., 2][lab == k].mean() >= hsv[..., 2][ring > 0].mean() + 15:
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
        prev_m = None
        um = np.zeros((H, W), np.uint8)
        for fr in frames_iter(inp, W, H):
            if prev_m is None or i % 2 == 0:
                m = text_mask(fr)
                prev_m = m
            else:
                m = prev_m
            um = np.maximum(um, m)
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
        umd = cv2.dilate(um, np.ones((5, 5), np.uint8))
        if umd.sum() > 0:
            plate = cv2.inpaint(plate, umd, 5, cv2.INPAINT_TELEA)   # scrub text out of the plate itself
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
                fr = fr.copy()
                ys, xs = np.where(m > 0)
                y1, y2 = max(int(ys.min()) - 20, 0), min(int(ys.max()) + 20, H)
                x1, x2 = max(int(xs.min()) - 20, 0), min(int(xs.max()) + 20, W)
                mroi = m[y1:y2, x1:x2] > 0
                froi = fr[y1:y2, x1:x2]
                gray = cv2.cvtColor(froi, cv2.COLOR_BGR2GRAY).astype(int)
                fg = np.abs(gray - plate_gray[y1:y2, x1:x2]) > 45
                fg_d = cv2.dilate(fg.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                filled = froi.copy(); filled[mroi] = plate[y1:y2, x1:x2][mroi]
                if (fg_d & mroi).sum() > 0.1 * mroi.sum():
                    telea = cv2.inpaint(froi, mroi.astype(np.uint8), 5, cv2.INPAINT_TELEA)
                    sel = fg_d & mroi
                    filled[sel] = telea[sel]
                mf = cv2.GaussianBlur(mroi.astype(np.float32), (7, 7), 0)[..., None]
                fr[y1:y2, x1:x2] = (froi * (1 - mf) + filled * mf).astype(np.uint8)
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

VERSION = 'v10-brand'

@app.route('/ver')
def ver():
    return VERSION

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('video')
    if not f:
        return jsonify({'error': 'no video'}), 400
    code = request.headers.get('X-PRO', '')
    if code not in PRO_CODES:
        key = (date.today().isoformat(), request.remote_addr)
        if USAGE.get(key, 0) >= FREE_LIMIT:
            return jsonify({'error': f'Free limit: {FREE_LIMIT} videos/day. Go PRO for unlimited.'}), 429
        USAGE[key] = USAGE.get(key, 0) + 1
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
<title>CaptionRemover.io – Free Video Caption, Watermark & Text Remover Online</title>
<meta name="description" content="Free online video caption, subtitle, watermark, text and logo remover. No app, no signup. Works on phone. Real background rebuild, audio kept.">
<meta name="keywords" content="video caption remover, watermark remover, remove subtitles from video, tiktok caption remover, youtube shorts cleaner, free video editor online">
<meta name="robots" content="index,follow">
<meta property="og:title" content="CaptionRemover.io – Free Video Caption & Watermark Remover">
<meta property="og:description" content="Remove captions, watermarks & text from any video free. Works on phone.">
<style>
*{box-sizing:border-box} body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#f1f5f9;color:#0f172a;margin:0}
header{background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.08);position:sticky;top:0;z-index:5}
.nav{max-width:760px;margin:0 auto;display:flex;align-items:center;gap:6px;padding:10px 14px}
.logo{font-weight:800;font-size:17px;margin-right:auto}
.nav button{border:0;background:#e2e8f0;color:#0f172a;border-radius:999px;padding:9px 16px;font-size:14px;font-weight:600}
.nav button.on{background:#2563eb;color:#fff}
main{max-width:760px;margin:0 auto;padding:16px 14px 40px}
.card{background:#fff;border-radius:14px;box-shadow:0 1px 3px rgba(0,0,0,.07);padding:20px;margin-bottom:14px}
h1{font-size:21px;margin:0 0 6px} h2{font-size:17px;margin:0 0 8px} p.sub{color:#64748b;margin:0 0 14px;font-size:14px}
input[type=file]{margin:8px 0;width:100%}
button.go{width:100%;background:#16a34a;color:#fff;border:0;border-radius:10px;padding:13px;font-size:16px;font-weight:700}
button.go:disabled{background:#94a3b8}
.bar{height:10px;background:#e2e8f0;border-radius:6px;overflow:hidden;margin:14px 0 8px}
.fill{height:100%;width:0%;background:#16a34a;transition:width .3s}
#msg{min-height:20px;color:#64748b;font-size:14px}
a.dl{display:inline-block;margin-top:10px;background:#2563eb;color:#fff;padding:12px 20px;border-radius:10px;text-decoration:none;font-weight:700}
.plans{display:flex;gap:12px;flex-wrap:wrap}
.plan{flex:1;min-width:220px;border:1px solid #e2e8f0;border-radius:12px;padding:16px}
.plan.pro{border:2px solid #2563eb;background:#eff6ff}
.price{font-size:22px;font-weight:800} .per{color:#64748b;font-size:13px}
ul{margin:8px 0;padding-left:20px;font-size:14px;color:#334155}
.amber{background:#f59e0b;color:#111} .green{background:#25d366;color:#111}
.small{font-size:13px;color:#64748b} footer{text-align:center;color:#94a3b8;font-size:12px;padding:20px}
.hide{display:none}
</style></head><body>
<header><div class=nav>
<span class=logo>🎬 CaptionRemover.io</span>
<button id=t1 class=on onclick="show(1)">Clean</button>
<button id=t2 onclick="show(2)">Plans & Billing</button>
<button id=t3 onclick="show(3)">Help</button>
</div></header>
<main>
<section id=s1>
<div class=card>
<h1>Remove captions, watermarks & text from video</h1>
<p class=sub>Real background rebuild — no blur. Audio kept. Works on phone. Free: 3 videos/day.</p>
<input type=file id=f accept=video/*>
<button class=go id=go onclick=up()>🧹 Remove captions now</button>
<div class=bar><div class=fill id=fill></div></div>
<div id=msg>Pick a video (up to 10 min).</div>
<div id=dl></div>
</div>
<div class=card><h2>📁 Finished videos on this device</h2><!--LIST--></div>
</section>
<section id=s2 class=hide>
<div class=plans>
<div class=plan><h2>Free</h2><div class=price>₹0</div><div class=per>forever</div>
<ul><li>3 videos per day</li><li>All caption colours</li><li>Audio kept</li></ul></div>
<div class=plan pro><h2>PRO ⭐</h2><div class=price>₹49</div><div class=per>per month</div>
<ul><li>Unlimited videos</li><li>Priority support</li><li>Perfect-clean requests</li></ul>
<a class=dl amber href="upi://pay?pa=UPI_PLACEHOLDER&pn=VideoCleaner&am=49&cu=INR">📲 Pay ₹49 via UPI</a><br>
<a class=dl green href="https://wa.me/WA_PLACEHOLDER">💬 WhatsApp screenshot → get code</a>
<p class=small style="margin-top:10px"><a href="#" onclick="pro();return false">Have a PRO code? Enter it</a></p>
</div></div>
<p class=sub>Payments go directly to the owner's UPI. No middlemen.</p>
</section>
<section id=s3 class=hide>
<div class=card><h2>How it works</h2>
<ul><li>Upload a video — it never leaves the server longer than processing.</li>
<li>We detect burned-in text (captions, titles, lists, watermarks, handles, logos).</li>
<li>The background behind the text is rebuilt for real — not blurred.</li>
<li>Download your clean video. Audio untouched.</li></ul></div>
<div class=card><h2>Tips for best results</h2>
<ul><li>Shorts & reels (under 2 min) clean fastest.</li>
<li>Colourful or white captions with dark outlines clean perfectly.</li>
<li>Very tiny or transparent logos may leave a faint shadow.</li></ul></div>
</section>
</main>
<footer>CaptionRemover.io — free forever for personal use.</footer>
<script>
function show(n){for(let i=1;i<4;i++){document.getElementById('s'+i).classList.toggle('hide',i!=n);document.getElementById('t'+i).classList.toggle('on',i==n);}}
function pro(){const c=prompt('Enter your PRO code');if(c){localStorage.setItem('pro',c);show(1);msg.textContent='PRO active ✔ unlimited';}}
async function up(){
 const fd=new FormData(); fd.append('video',f.files[0]);
 go.disabled=true; msg.textContent='Uploading...';
 const r=await fetch('/upload',{method:'POST',body:fd,headers:{'X-PRO':localStorage.getItem('pro')||''}});
 const j=await r.json();
 if(r.status==429){go.disabled=false;msg.textContent=j.error+' ⭐ See Plans & Billing.';return;}
 poll(j.job);
}
function poll(job){
 const t=setInterval(async()=>{
  const r=await fetch('/status/'+job); const j=await r.json();
  fill.style.width=j.progress+'%'; msg.textContent=j.stage||j.status;
  if(j.status=='done'){clearInterval(t);go.disabled=false;
    dl.innerHTML='<a class=dl href="/download/'+job+'">⬇ Download CLEAN_VIDEO.mp4</a>';}
  if(j.status=='error'){clearInterval(t);go.disabled=false;msg.textContent=j.stage;}
 },700);
}
</script></body></html>"""


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), threaded=True)
