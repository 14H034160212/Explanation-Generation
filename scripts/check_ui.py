import urllib.request
import json
import websocket
import time

def send_msg(ws, method, params=None):
    msg = {"id": int(time.time() * 1000) % 1000000, "method": method}
    if params:
        msg["params"] = params
    ws.send(json.dumps(msg))
    
    ws.settimeout(2.0)
    start = time.time()
    while time.time() - start < 5:
        try:
            resp = json.loads(ws.recv())
            if "id" in resp and resp["id"] == msg["id"]:
                return resp.get("result", resp)
        except:
            pass
    return None

def main():
    req = urllib.request.Request("http://localhost:9222/json")
    with urllib.request.urlopen(req) as response:
        pages = json.loads(response.read().decode())
    
    ws_url = None
    for page in pages:
        if "overleaf" in page.get("url", "").lower() and "project" in page.get("url", "").lower() and page.get("type", "") == "page":
            ws_url = page.get("webSocketDebuggerUrl")
            break
            
    if not ws_url:
        print("Not found Overleaf tab")
        return
        
    ws = websocket.create_connection(ws_url, suppress_origin=True)
    send_msg(ws, "Runtime.enable")
    
    # Let's run a broad selector to see all files in the tree
    js = """
    (() => {
        let items = Array.from(document.querySelectorAll('*'))
            .filter(el => el.textContent && (el.textContent.includes('.tex') || el.textContent.includes('main.tex')));
        // filter out huge parent containers by checking text length
        items = items.filter(el => el.textContent.length < 50);
        return items.map(el => el.tagName + '#' + el.id + '.' + el.className + ' : ' + el.textContent.trim()).filter((v, i, a) => a.indexOf(v) === i).slice(0, 10).join('\\n');
    })();
    """
    res = send_msg(ws, "Runtime.evaluate", {"expression": js, "returnByValue": True})
    print("Files found in UI:\n", res)
    
if __name__ == "__main__":
    main()
