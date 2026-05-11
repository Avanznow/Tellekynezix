# Python 2/3 Coexistence Solution - Quick Reference

## 🎯 Problem Solved

**Challenge:** Avatar project requires both Python 2.7 (NAO6 robot via NAOqi SDK) and Python 3.x (Tello drone + GUI)  
**Solution:** Microservice architecture with HTTP REST API communication

---

## 📁 Solution Files

| File | Purpose |
|------|---------|
| `README.md` | Complete documentation |
| `start_nao_service.sh` | Launch NAO Python 2.7 service |
| `start_tello_environment.sh` | Launch Tello/Avatar Python 3 environment |
| `start_avatar_full.sh` | Launch BOTH services together |
| `install_python2.sh` | Install Python 2.7 (Ubuntu/Debian) |
| `install_python2_macos.sh` | Install Python 2.7 (macOS) |
| `test_environment.py` | Verify Python 3 setup |
| `test_nao_service.py` | Test NAO service connectivity |
| `nao_service_requirements.txt` | Python 2.7 dependencies |

---

## 🚀 Quick Start (3 Steps)

### 1. Install Python 2.7
```bash
# Ubuntu/Debian
./install_python2.sh

# macOS
./install_python2_macos.sh
```

### 2. Extract NAOqi SDK
```bash
cd /path/to/Avatar
tar -xzf pynaoqi-python2.7-*.tar.gz
```

### 3. Launch Full Environment
```bash
cd python_2and3
./start_avatar_full.sh
```

---

## 🔍 Architecture

```
┌─────────────────────────────────────┐
│      Avatar GUI (Python 3)          │
│  - Tello Drone Control              │
│  - BrainFlow Processing             │
│  - PySide6 Interface                │
└─────────────┬───────────────────────┘
              │
              │ HTTP REST API
              │ (localhost:5000)
              │
┌─────────────▼───────────────────────┐
│    NAO Service (Python 2.7)         │
│  - Flask HTTP Server                │
│  - NAOqi SDK Integration            │
│  - NAO6 Robot Control               │
└─────────────────────────────────────┘
```

---

## 📡 NAO Service API Endpoints

### Health Check
```bash
curl http://localhost:5000/health
```

### Connect to NAO
```bash
curl -X POST http://localhost:5000/api/connect \
  -H "Content-Type: application/json" \
  -d '{"ip": "192.168.1.100", "port": 9559}'
```

### Execute Command
```bash
curl -X POST http://localhost:5000/api/command \
  -H "Content-Type: application/json" \
  -d '{"action": "stand_up"}'
```

**Available Commands:**  
`connect`, `stand_up`, `sit_down`, `walk_forward`, `walk_backward`, `turn_left`, `turn_right`, `wave`, `say`

---

## ✅ Testing & Verification

### Test Python 3 Environment
```bash
python3 test_environment.py
```

### Test NAO Service
```bash
# Start NAO service first
./start_nao_service.sh

# In another terminal
python3 test_nao_service.py
```

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| Python 2.7 not found | Run `./install_python2.sh` or `./install_python2_macos.sh` |
| NAOqi SDK not detected | Extract SDK to project root |
| Port 5000 in use | `lsof -ti:5000 \| xargs kill -9` |
| Flask import error | `python2.7 -m pip install flask==1.1.4 werkzeug==1.0.1` |
| Avatar can't connect to NAO | Check `curl http://localhost:5000/health` |

---

## 📊 Advantages of This Solution

✅ **Process Isolation** - No dependency conflicts  
✅ **Simple Integration** - Standard HTTP/REST  
✅ **Easy Testing** - Test each service independently  
✅ **Future-Proof** - Easy migration path when NAOqi supports Python 3  
✅ **Clear Separation** - Each environment has its own packages  

---

## 🔐 Security Notes

- NAO service listens on `localhost` only (no external exposure)
- For production: Add authentication, HTTPS, rate limiting
- Python 2.7 reached EOL in 2020 - isolate NAO service

---

## 📞 Quick Help

**NAO service won't start?**  
→ Check `tail -f nao_service.log`

**Tello can't connect?**  
→ Verify WiFi connection to Tello-XXXXXX

**GUI won't launch?**  
→ Run `python3 test_environment.py` to check dependencies

---

**Last Updated:** 2026-05-10  
**Version:** 1.0  
**Issue:** Closes #9
