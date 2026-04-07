document.addEventListener('DOMContentLoaded', () => {
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');
    const tabNameDisplay = document.getElementById('current-tab-name');

    // Tab Switching Logic
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const tabId = item.getAttribute('data-tab');

            // UI Update
            navItems.forEach(n => n.classList.remove('active'));
            tabContents.forEach(t => t.classList.remove('active'));

            item.classList.add('active');
            document.getElementById(`${tabId}-tab`).classList.add('active');
            tabNameDisplay.textContent = item.textContent;
        });
    });

    // --- Authentication Logic ---
    const loginModal = document.getElementById('login-modal');
    const loginBtn = document.getElementById('login-btn');
    const loginError = document.getElementById('login-error');

    // Check if already logged in
    if (localStorage.getItem('v2_token')) {
        loginModal.style.display = 'none';
    }

    loginBtn.addEventListener('click', async () => {
        const tenant_id = document.getElementById('login-tenant').value;
        const password = document.getElementById('login-password').value;

        try {
            const resp = await fetch('/v1/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tenant_id, password })
            });

            if (!resp.ok) throw new Error('Invalid credentials');
            const data = await resp.json();
            localStorage.setItem('v2_token', data.access_token);
            loginModal.style.display = 'none';
        } catch (err) {
            loginError.textContent = err.message;
            loginError.style.display = 'block';
        }
    });

    // --- Key Rotation Logic ---
    const rotateBtn = document.getElementById('rotate-key-btn');
    const newKeyArea = document.getElementById('new-key-area');
    const newKeyVal = document.getElementById('new-api-key-val');

    rotateBtn.addEventListener('click', async () => {
        if (!confirm('Are you sure you want to rotate your API key?')) return;

        try {
            const resp = await fetch('/v1/auth/rotate-key', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${localStorage.getItem('v2_token')}`
                }
            });

            if (!resp.ok) throw new Error('Failed to rotate key');
            const data = await resp.json();

            newKeyVal.textContent = data.api_key;
            newKeyArea.style.display = 'block';
        } catch (err) {
            alert(err.message);
        }
    });

    // Chat Interaction
    const sendBtn = document.getElementById('send-btn');
    // ... rest of the file
    const userInput = document.getElementById('user-input');
    const messagesContainer = document.getElementById('messages');

    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        // Add User Message
        const userDiv = document.createElement('div');
        userDiv.className = 'message user glass';
        userDiv.innerHTML = `<p>${text}</p>`;
        messagesContainer.appendChild(userDiv);
        userInput.value = '';
        messagesContainer.scrollTop = messagesContainer.scrollHeight;

        // Add Loading State
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'message assistant glass loading';
        loadingDiv.innerHTML = `<p>V2 Engine is generating (Multi-query search + RRF)...</p>`;
        messagesContainer.appendChild(loadingDiv);

        try {
            const response = await fetch(`/v1/tenants/demo/rag/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: [{ role: 'user', content: text }],
                    stream: false,
                    temperature: 0.2
                })
            });

            if (!response.ok) throw new Error('API Error');
            const data = await response.json();

            loadingDiv.remove();

            const assistantDiv = document.createElement('div');
            assistantDiv.className = 'message assistant glass';
            assistantDiv.innerHTML = `<p>${data.choices[0].message.content}</p>`;
            messagesContainer.appendChild(assistantDiv);

            // If it's a SQL request, update the SQL tab
            if (text.toLowerCase().includes('sql') || text.toLowerCase().includes('query')) {
                document.getElementById('sql-output').textContent = data.choices[0].message.content;
            }

        } catch (err) {
            loadingDiv.innerHTML = `<p style="color: #ef4444">Error: ${err.message}</p>`;
        }

        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
});
