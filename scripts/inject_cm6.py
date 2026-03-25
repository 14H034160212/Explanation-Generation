import urllib.request
import json
import websocket
import time
import sys

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
        except websocket.WebSocketTimeoutException:
            pass
        except Exception as e:
            pass
    return None

def main(target_file, local_file):
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
    
    with open(local_file, "r") as f:
        text = f.read()

    # Click the file in the sidebar
    print(f"Selecting {target_file} in sidebar...")
    click_js = f"""
    (() => {{
        let spans = Array.from(document.querySelectorAll('span, div.item-name-button, div.entity-name'));
        let fileBtn = spans.find(s => s.textContent.trim() === '{target_file}');
        if (fileBtn) {{
            fileBtn.click();
            let parent = fileBtn.closest('li') || fileBtn.closest('div.entity');
            if (parent && parent !== fileBtn) parent.click();
            return "Clicked " + '{target_file}';
        }}
        return "File not found: " + '{target_file}';
    }})();
    """
    res = send_msg(ws, "Runtime.evaluate", {"expression": click_js, "returnByValue": True})
    print(res)
    time.sleep(2)
    
    print("Focusing editor...")
    focus_js = """
    (() => {
        let el = document.querySelector('.cm-content');
        if (el) { el.focus(); return "Focused"; }
        return "Not found";
    })();
    """
    send_msg(ws, "Runtime.evaluate", {"expression": focus_js, "returnByValue": True})
    time.sleep(1)
    
    print("Selecting all & deleting...")
    send_msg(ws, "Input.dispatchKeyEvent", {"type": "rawKeyDown", "windowsVirtualKeyCode": 17, "key": "Control"})
    send_msg(ws, "Input.dispatchKeyEvent", {"type": "keyDown", "windowsVirtualKeyCode": 65, "key": "a"})
    send_msg(ws, "Input.dispatchKeyEvent", {"type": "keyUp", "windowsVirtualKeyCode": 65, "key": "a"})
    send_msg(ws, "Input.dispatchKeyEvent", {"type": "keyUp", "windowsVirtualKeyCode": 17, "key": "Control"})
    time.sleep(0.5)
    
    send_msg(ws, "Input.dispatchKeyEvent", {"type": "keyDown", "windowsVirtualKeyCode": 8, "key": "Backspace"})
    send_msg(ws, "Input.dispatchKeyEvent", {"type": "keyUp", "windowsVirtualKeyCode": 8, "key": "Backspace"})
    time.sleep(0.5)
    
    print(f"Injecting {len(text)} characters...")
    chunk_size = 500
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        send_msg(ws, "Input.insertText", {"text": chunk})
        time.sleep(0.05)
        
    print("Done injecting", target_file)

    print("Recompiling...")
    recompile_js = """
    (() => {
        let recompileBtn = Array.from(document.querySelectorAll('button')).find(b => b.textContent && b.textContent.includes('Recompile'));
        if (recompileBtn) { recompileBtn.click(); return "Recompiling"; }
        return "Not found Recompile";
    })();
    """
    res = send_msg(ws, "Runtime.evaluate", {"expression": recompile_js, "returnByValue": True})
    print(res)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
