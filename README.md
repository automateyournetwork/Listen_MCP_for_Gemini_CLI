# Listen_MCP_for_Gemini_CLI
An MCP for /listen custom slash commands avoiding shell locking to allow for in-line server experience

### There is nothign to do here ###
You will add this to your Gemini settings.json file as such: 

    "listen": {
      "command": "/home/<homedir>/Listen_MCP_for_Gemini_CLI/venv/bin/python",
      "args": [
        "/home/<homedir>/Listen_MCP_for_Gemini_CLI/listen.py"
      ]
    }

As you see you need to setup a virtual environment and install the requirements.txt file in this MCP directory for it work.

python3 -m pip install -r requirements.txt

