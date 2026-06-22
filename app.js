// Frontend logic for AetherPDF

// Global variables
let activeFileHash = null;
let activeFileName = "";
let chatHistory = []; // stores [{role: "user"/"assistant", content: "..."}]
let docMode = "strict"; // "strict" or "open"
let splashTimeout = null;

// DOM Elements
const sections = {
    landing: document.getElementById("landing-page"),
    upload: document.getElementById("upload-page"),
    loading: document.getElementById("loading-page"),
    dashboard: document.getElementById("dashboard-page")
};

const navBtns = {
    upload: document.getElementById("nav-upload-btn"),
    help: document.getElementById("nav-help-btn"),
    logo: document.getElementById("logo-btn")
};

const closePdfBtn = document.getElementById("close-pdf-btn");
const confirmModal = document.getElementById("confirm-modal");
const confirmYesBtn = document.getElementById("confirm-yes-btn");
const confirmCancelBtn = document.getElementById("confirm-cancel-btn");

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const errorMessage = document.getElementById("error-message");
const errorText = document.getElementById("error-text");

// Loading elements
const loadingTitle = document.getElementById("loading-title");
const loadingStatus = document.getElementById("loading-status");
const progressFill = document.getElementById("progress-fill");

// Dashboard elements
const pdfIframe = document.getElementById("pdf-iframe");
const viewerFileName = document.getElementById("viewer-file-name");
const metaSize = document.getElementById("meta-size");
const metaPages = document.getElementById("meta-pages");
const metaChars = document.getElementById("meta-chars");
const modeCheckbox = document.getElementById("mode-checkbox");
const modeBadge = document.getElementById("mode-badge");
const modeDesc = document.getElementById("mode-desc");
const summaryText = document.getElementById("summary-text");

// Chat elements
const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const clearChatBtn = document.getElementById("clear-chat-btn");

// Help Modal elements
const helpModal = document.getElementById("help-modal");
const closeModalBtns = document.querySelectorAll(".close-modal-btn, .close-modal-btn-action");

// ---------------------------------------------------------------------
// 1. SPA Routing & State Management
// ---------------------------------------------------------------------

function triggerSplashTransition() {
    if (splashTimeout) {
        clearTimeout(splashTimeout);
    }
    
    const landing = document.getElementById("landing-page");
    if (!landing) return;
    
    landing.classList.remove("fade-out");
    
    // Auto-transition: wait 4.2 seconds (plus 0.8s fade-out = 5 seconds total), then start fading out
    splashTimeout = setTimeout(() => {
        landing.classList.add("fade-out");
        
        // Wait 800ms for CSS fade-out transition to complete, then change SPA view
        splashTimeout = setTimeout(() => {
            navigateTo("upload-page");
        }, 800);
    }, 4200);
}

function navigateTo(sectionId) {
    // Hide all sections, show active one
    Object.keys(sections).forEach(key => {
        if (sections[key].id === sectionId) {
            sections[key].classList.add("active");
        } else {
            sections[key].classList.remove("active");
        }
    });

    // Handle active state of nav links
    if (sectionId === "upload-page") {
        navBtns.upload.classList.add("active");
    } else {
        navBtns.upload.classList.remove("active");
    }

    // Toggle navbar visibility and main content spacing
    const navbar = document.querySelector(".navbar");
    const mainContent = document.querySelector(".main-content");
    if (sectionId === "landing-page") {
        navbar.classList.add("hidden");
        mainContent.classList.add("no-navbar");
        triggerSplashTransition();
    } else {
        navbar.classList.remove("hidden");
        mainContent.classList.remove("no-navbar");
        
        // Cancel splash timeout if navigated away early
        if (splashTimeout) {
            clearTimeout(splashTimeout);
        }
    }
}

// Reset app state and go back to upload
function resetAndGoToUpload() {
    activeFileHash = null;
    activeFileName = "";
    chatHistory = [];
    pdfIframe.src = "";
    summaryText.innerHTML = "";
    
    // Clear chat bubbles to only default greeting
    chatMessages.innerHTML = `
        <div class="chat-message bot">
            <div class="message-bubble">
                Hello! I have processed this document. You can ask me questions about it, request sections to be explained, or switch to "Open" mode to expand on external topics.
            </div>
        </div>
    `;
    
    chatInput.value = "";
    sendBtn.disabled = true;
    navigateTo("upload-page");
}

// Function to handle closing active document (shows confirmation modal if active)
function handleCloseDocument(e) {
    if (e) e.preventDefault();
    if (activeFileHash) {
        confirmModal.classList.add("active");
    } else {
        navigateTo("upload-page");
    }
}

// Event Listeners for Navigation
navBtns.upload.addEventListener("click", handleCloseDocument);
closePdfBtn.addEventListener("click", handleCloseDocument);

navBtns.logo.addEventListener("click", (e) => {
    e.preventDefault();
    navigateTo("landing-page");
});

// Confirmation Modal Controls
confirmYesBtn.addEventListener("click", () => {
    confirmModal.classList.remove("active");
    resetAndGoToUpload();
});

confirmCancelBtn.addEventListener("click", () => {
    confirmModal.classList.remove("active");
});

confirmModal.addEventListener("click", (e) => {
    if (e.target === confirmModal) {
        confirmModal.classList.remove("active");
    }
});

// Help Modal Controls
navBtns.help.addEventListener("click", () => {
    helpModal.classList.add("active");
});

closeModalBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        helpModal.classList.remove("active");
    });
});

// Close modal if user clicks outside container
helpModal.addEventListener("click", (e) => {
    if (e.target === helpModal) {
        helpModal.classList.remove("active");
    }
});

// ---------------------------------------------------------------------
// 2. Drag & Drop File Upload Handling
// ---------------------------------------------------------------------

// Click zone to browse
dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
    }
});

// Dragover styling
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
        handleFileSelection(e.dataTransfer.files[0]);
    }
});

function handleFileSelection(file) {
    errorMessage.classList.add("hidden");
    
    // 1. Validate File extension
    if (!file.name.lowerCase?.endsWith(".pdf") && !file.name.endsWith(".pdf") && !file.name.endsWith(".PDF")) {
        showError("Invalid file type. Only PDF (.pdf) files are supported.");
        return;
    }
    
    // 2. Validate File Size (50 MB limit)
    const maxSizeBytes = 50 * 1024 * 1024;
    if (file.size > maxSizeBytes) {
        showError("File is too large. Maximum supported PDF size is 50 MB.");
        return;
    }
    
    // Proceed to upload & processing
    activeFileName = file.name;
    uploadFile(file);
}

function showError(msg) {
    errorText.innerText = msg;
    errorMessage.classList.remove("hidden");
}

// ---------------------------------------------------------------------
// 3. Backend Communication - Upload, Summarize, Stream Q&A
// ---------------------------------------------------------------------

async function uploadFile(file) {
    navigateTo("loading-page");
    loadingTitle.innerText = "Ingesting PDF";
    loadingStatus.innerText = "Analyzing file contents...";
    progressFill.style.width = "15%";
    
    const formData = new FormData();
    formData.append("file", file);
    
    try {
        progressFill.style.width = "40%";
        loadingStatus.innerText = "Extracting text structure (and running OCR fallback if scanned)...";
        
        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || "Failed to process PDF.");
        }
        
        progressFill.style.width = "85%";
        loadingStatus.innerText = "Generating local semantic embeddings...";
        
        const data = await response.json();
        activeFileHash = data.file_hash;
        
        progressFill.style.width = "100%";
        loadingStatus.innerText = "Initialization complete!";
        
        // Transition to Dashboard
        setTimeout(() => {
            initializeDashboard(data);
        }, 500);
        
    } catch (err) {
        console.error(err);
        navigateTo("upload-page");
        showError(err.message || "An error occurred during file upload.");
    }
}

function initializeDashboard(metadata) {
    // 1. Set Viewer PDF
    pdfIframe.src = `/pdf/${metadata.file_hash}`;
    viewerFileName.innerText = metadata.filename;
    
    // 2. Set Metadata Card
    const sizeMb = (metadata.char_count * 2) / (1024 * 1024); // mock calculation or get it
    metaSize.innerText = `${(metadata.char_count / 150000).toFixed(2)} MB`; // rough display file estimate if not returned
    metaPages.innerText = metadata.page_count;
    metaChars.innerText = metadata.char_count.toLocaleString();
    
    // If it's a cached document or has physical details, we fetch size if needed
    // The sizes are estimates for display, pages and chars are exact from parser.
    
    // 3. Clear existing states
    summaryText.innerHTML = `
        <div class="pulse-text-loader">
            Generating document summary... <span class="dot-bounce">.</span><span class="dot-bounce">.</span><span class="dot-bounce">.</span>
        </div>
    `;
    
    navigateTo("dashboard-page");
    
    // 4. Trigger Summarization (Async Streaming)
    generateSummary(metadata.file_hash);
}

async function generateSummary(fileHash) {
    try {
        const response = await fetch("/summarize", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ file_hash: fileHash })
        });
        
        if (!response.ok) {
            summaryText.innerText = "Unable to generate summary for this document.";
            return;
        }
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let done = false;
        let fullText = "";
        let firstChunk = true;
        
        while (!done) {
            const { value, done: readerDone } = await reader.read();
            done = readerDone;
            const chunk = decoder.decode(value, { stream: !done });
            fullText += chunk;
            
            if (fullText.trim()) {
                if (firstChunk) {
                    summaryText.innerHTML = ""; // Clear "Generating..." loader on first text chunk
                    firstChunk = false;
                }
                summaryText.innerHTML = marked.parse(fullText);
            }
        }
    } catch (err) {
        console.error(err);
        summaryText.innerText = "Error loading document summary.";
    }
}

// ---------------------------------------------------------------------
// 4. Document Answering Mode Toggle Switch
// ---------------------------------------------------------------------

modeCheckbox.addEventListener("change", (e) => {
    if (e.target.checked) {
        docMode = "open";
        modeBadge.innerText = "OPEN";
        modeBadge.classList.add("open-badge");
        modeDesc.innerText = "Uses the PDF as base context, but LLM can draw from general knowledge to explain terms.";
    } else {
        docMode = "strict";
        modeBadge.innerText = "STRICT";
        modeBadge.classList.remove("open-badge");
        modeDesc.innerText = "Replies are strictly bounded to the PDF context. Hallucination-free.";
    }
});

// ---------------------------------------------------------------------
// 5. Chat Box (Streaming & History Memory)
// ---------------------------------------------------------------------

// Input styling & send trigger
chatInput.addEventListener("input", () => {
    sendBtn.disabled = chatInput.value.trim().length === 0;
    
    // Auto-expand input height slightly
    chatInput.style.height = "auto";
    chatInput.style.height = (chatInput.scrollHeight) + "px";
    if (chatInput.scrollHeight > 150) {
        chatInput.style.overflowY = "scroll";
        chatInput.style.height = "150px";
    } else {
        chatInput.style.overflowY = "hidden";
    }
});

chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (chatInput.value.trim().length > 0) {
            handleSendMessage();
        }
    }
});

sendBtn.addEventListener("click", handleSendMessage);

clearChatBtn.addEventListener("click", () => {
    chatHistory = [];
    chatMessages.innerHTML = `
        <div class="chat-message bot">
            <div class="message-bubble">
                Chat history cleared. You can ask me new questions about the PDF.
            </div>
        </div>
    `;
});

async function handleSendMessage() {
    const query = chatInput.value.trim();
    if (!query) return;
    
    // 1. Add User message bubble to screen
    appendChatBubble("user", query);
    
    // Clear input
    chatInput.value = "";
    chatInput.style.height = "auto";
    sendBtn.disabled = true;
    
    // 2. Create Placeholder Bot message bubble
    const botBubble = appendChatBubble("bot", "");
    const bubbleContent = botBubble.querySelector(".message-bubble");
    bubbleContent.innerHTML = `
        <div class="pulse-text-loader">
            Thinking <span class="dot-bounce">.</span><span class="dot-bounce">.</span><span class="dot-bounce">.</span>
        </div>
    `;
    
    try {
        const response = await fetch("/query", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                file_hash: activeFileHash,
                query: query,
                doc_mode: docMode,
                chat_history: chatHistory
            })
        });
        
        if (!response.ok) {
            bubbleContent.innerText = "Error: Failed to fetch response from model.";
            return;
        }
        
        bubbleContent.innerHTML = ""; // clear loader
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let done = false;
        let finalResponseText = "";
        
        while (!done) {
            const { value, done: readerDone } = await reader.read();
            done = readerDone;
            const chunk = decoder.decode(value, { stream: !done });
            finalResponseText += chunk;
            bubbleContent.innerHTML = marked.parse(finalResponseText);
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        // Append query and answer to local chat memory
        chatHistory.push({"role": "user", "content": query});
        chatHistory.push({"role": "assistant", "content": finalResponseText});
        
    } catch (err) {
        console.error(err);
        bubbleContent.innerText = "Connection lost. Please verify that the backend server is running.";
    }
}

function appendChatBubble(sender, text) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("chat-message", sender);
    
    const bubbleDiv = document.createElement("div");
    bubbleDiv.classList.add("message-bubble");
    bubbleDiv.innerText = text;
    
    messageDiv.appendChild(bubbleDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll chat window to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageDiv;
}

// Initialize application state
document.addEventListener("DOMContentLoaded", () => {
    navigateTo("landing-page");
});
