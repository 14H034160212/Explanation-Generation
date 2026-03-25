import urllib.request
import json
import websocket
import time

def send_msg(ws, method, params=None):
    msg = {"id": int(time.time() * 1000) % 1000000, "method": method}
    if params:
        msg["params"] = params
    ws.send(json.dumps(msg))
    
    ws.settimeout(15.0)
    start = time.time()
    while time.time() - start < 20:
        try:
            resp = json.loads(ws.recv())
            if "id" in resp and resp["id"] == msg["id"]:
                return resp.get("result", resp)
        except websocket.WebSocketTimeoutException:
            pass
        except Exception as e:
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
            print("Found:", page.get("url"))
            break
            
    if not ws_url:
        print("Not found")
        return
        
    ws = websocket.create_connection(ws_url, suppress_origin=True)
    send_msg(ws, "DOM.enable")
    send_msg(ws, "Runtime.enable")
    
    # Text processing
    results_tex = ""
    with open("paper_draft/results.tex", "r") as f:
        results_tex += f.read()
    results_tex += "\n\n"
    with open("paper_draft/extra_results.tex", "r") as f:
        results_tex += f.read()
        
    main_tex = ""
    with open("paper_draft/main.tex", "r") as f:
        for line in f:
            if "extra_results" not in line:
                main_tex += line

    def click_file(name):
        js = f"""
        (() => {{
            let spans = Array.from(document.querySelectorAll('.file-tree-container li span, .file-tree-container div'));
            let fileBtn = spans.find(s => s.textContent.trim() === '{name}');
            if (fileBtn) {{
                fileBtn.click();
                let parent = fileBtn.closest('li') || fileBtn.closest('div');
                if (parent && parent !== fileBtn) parent.click();
                return "Clicked " + '{name}';
            }}
            return "File not found: " + '{name}';
        }})();
        """
        res = send_msg(ws, "Runtime.evaluate", {"expression": js, "returnByValue": True})
        print(res)

    def insert_text(text):
        # 1. Focus the editor window
        focus_js = """
        (() => {
            let el = document.querySelector('.cm-content');
            if (el) { el.focus(); return "Focused"; }
            return "Not found";
        })();
        """
        send_msg(ws, "Runtime.evaluate", {"expression": focus_js, "returnByValue": True})
        time.sleep(1)
        
        # 2. Key events to Select All
        send_msg(ws, "Input.dispatchKeyEvent", {"type": "rawKeyDown", "windowsVirtualKeyCode": 17, "key": "Control"})
        send_msg(ws, "Input.dispatchKeyEvent", {"type": "keyDown", "windowsVirtualKeyCode": 65, "key": "a"})
        send_msg(ws, "Input.dispatchKeyEvent", {"type": "keyUp", "windowsVirtualKeyCode": 65, "key": "a"})
        send_msg(ws, "Input.dispatchKeyEvent", {"type": "keyUp", "windowsVirtualKeyCode": 17, "key": "Control"})
        time.sleep(0.5)
        
        # Delete the selected text
        send_msg(ws, "Input.dispatchKeyEvent", {"type": "keyDown", "windowsVirtualKeyCode": 8, "key": "Backspace"})
        send_msg(ws, "Input.dispatchKeyEvent", {"type": "keyUp", "windowsVirtualKeyCode": 8, "key": "Backspace"})
        time.sleep(0.5)
        
        # 3. Use Input.insertText which natively simulates input typing over the focused node
        print("Inserting text length:", len(text))
        send_msg(ws, "Input.insertText", {"text": text})

    # --- ACTION ---
    print("Switching to results.tex...")
    click_file("results.tex")
    time.sleep(2)
    insert_text(results_tex)
    time.sleep(2)
    
    print("Switching to main.tex...")
    click_file("main.tex")
    time.sleep(2)
    insert_text(main_tex)
    time.sleep(2)
    
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
    print("Done")

if __name__ == "__main__":
    main()
