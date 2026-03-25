import urllib.request
import json
import websocket
import time
import os

def send_msg(ws, method, params=None):
    msg = {"id": int(time.time() * 1000) % 1000000, "method": method}
    if params:
        msg["params"] = params
    ws.send(json.dumps(msg))
    
    ws.settimeout(2.0)
    start = time.time()
    while time.time() - start < 10:
        try:
            resp = json.loads(ws.recv())
            if "id" in resp and resp["id"] == msg["id"]:
                return resp.get("result", resp)
        except websocket.WebSocketTimeoutException:
            pass
        except Exception as e:
            print("Error recv:", e)
            break
    return None

def main():
    req = urllib.request.Request("http://localhost:9222/json")
    with urllib.request.urlopen(req) as response:
        pages = json.loads(response.read().decode())
    
    ws_url = None
    for page in pages:
        if "overleaf" in page.get("url", "").lower() and "project" in page.get("url", "").lower() and page.get("type", "") == "page":
            ws_url = page.get("webSocketDebuggerUrl")
            print("Found Overleaf project tab:", page.get("url"))
            break
            
    if not ws_url:
        print("No open overleaf project found.")
        return
        
    ws = websocket.create_connection(ws_url, suppress_origin=True)
    send_msg(ws, "DOM.enable")
    send_msg(ws, "Runtime.enable")
    
    print("Injecting custom file input...")
    inject_js = """
    (() => {
        let inp = document.getElementById('hax_upload');
        if (!inp) {
            inp = document.createElement('input');
            inp.type = 'file';
            inp.id = 'hax_upload';
            inp.style.display = 'block';
            inp.style.position = 'fixed';
            inp.style.top = '0';
            inp.style.left = '0';
            inp.style.zIndex = '999999';
            inp.style.opacity = '0.01';
            document.body.appendChild(inp);
        }
        return true;
    })();
    """
    send_msg(ws, "Runtime.evaluate", {"expression": inject_js, "returnByValue": True})
    time.sleep(1)
    
    print("Getting node id for custom input via DOM.querySelector...")
    doc = send_msg(ws, "DOM.getDocument")
    root_node_id = doc['root']['nodeId']
    
    node = send_msg(ws, "DOM.querySelector", {"nodeId": root_node_id, "selector": "#hax_upload"})
    file_input_id = node["nodeId"]
    
    zip_path = os.path.abspath("paper_draft.zip")
    print("Uploading zip via CDP to custom input:", zip_path)
    res3 = send_msg(ws, "DOM.setFileInputFiles", {"files": [zip_path], "nodeId": file_input_id})
    print("DOM.setFileInputFiles result:", res3)
    time.sleep(1)
    
    print("Dispatching synthetic drop event...")
    drop_js = """
    (() => {
        let inp = document.getElementById('hax_upload');
        if (!inp.files || inp.files.length === 0) return "No files in input!";
        
        let file = inp.files[0];
        let dt = new DataTransfer();
        dt.items.add(file);
        
        let dropEvent = new DragEvent('drop', {
            bubbles: true,
            cancelable: true,
            dataTransfer: dt
        });
        
        // Find the main upload dropzone or just dispatch to window/body
        // Overleaf listens for drop events on window or specific containers.
        // Let's dispatch it to the document body to trigger global drop handlers.
        document.body.dispatchEvent(dropEvent);
        
        // Also dispatch to the file tree pane just in case
        let fileTree = document.querySelector('.file-tree-container') || document.body;
        fileTree.dispatchEvent(new DragEvent('drop', {
            bubbles: true,
            cancelable: true,
            dataTransfer: dt
        }));
        
        return "Dispatched drop event with zip file!";
    })();
    """
    res_drop = send_msg(ws, "Runtime.evaluate", {"expression": drop_js, "returnByValue": True})
    print("Drop JS outcome:", res_drop)
    
    print("Checking for overwrite modal...")
    time.sleep(2)
    confirm_js = """
    (() => {
        let btns = Array.from(document.querySelectorAll('.modal-footer button, .modal button'));
        let confirmBtn = btns.find(b => {
           let t = b.textContent.toLowerCase();
           return t.includes('overwrite') || t.includes('replace') || t.includes('confirm') || t.includes('upload');
        });
        if (confirmBtn) {
            confirmBtn.click();
            return "Confirmed modal!";
        }
        return "No overwrite modal currently.";
    })();
    """
    res_conf = send_msg(ws, "Runtime.evaluate", {"expression": confirm_js, "returnByValue": True})
    print("Confirm outcome:", res_conf)
    print("Done!")

if __name__ == "__main__":
    main()
