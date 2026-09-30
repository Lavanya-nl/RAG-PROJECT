const fileInput = document.querySelector('#file-input');
const uploadForm = document.querySelector('#upload-form');
const uploadStatus = document.querySelector('#upload-status');
const documentList = document.querySelector('#document-list');
const emptyLibrary = document.querySelector('#empty-library');
const documentCount = document.querySelector('#document-count');
const statusBox = document.querySelector('#model-status');
const statusLabel = document.querySelector('#status-label');
const refreshButton = document.querySelector('#refresh-status');
const chatForm = document.querySelector('#chat-form');
const questionInput = document.querySelector('#question-input');
const sendButton = document.querySelector('#send-button');
const messageList = document.querySelector('#message-list');
const welcome = document.querySelector('#welcome');

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = response.status === 204 ? null : await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload?.detail || `Request failed (${response.status}).`);
  return payload;
}

function setUploadStatus(message, isError = false) {
  uploadStatus.textContent = message;
  uploadStatus.classList.toggle('error', isError);
}

async function refreshDocuments() {
  const documents = await api('/api/documents');
  documentList.replaceChildren();
  emptyLibrary.hidden = documents.length > 0;
  documentCount.textContent = String(documents.length);

  for (const document of documents) {
    const item = document.createElement('li');
    item.className = 'document-item';

    const glyph = document.createElement('span');
    glyph.className = 'file-glyph';
    glyph.textContent = document.filename.toLowerCase().endsWith('.pdf') ? 'PDF' : 'DOCX';

    const copy = document.createElement('div');
    copy.className = 'document-copy';
    const name = document.createElement('div');
    name.className = 'document-name';
    name.textContent = document.filename;
    name.title = document.filename;
    const meta = document.createElement('div');
    meta.className = 'document-meta';
    meta.textContent = `${document.chunk_count} ${document.chunk_count === 1 ? 'chunk' : 'chunks'}`;
    copy.append(name, meta);

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'remove-button';
    remove.textContent = '×';
    remove.setAttribute('aria-label', `Remove ${document.filename}`);
    remove.title = 'Remove document';
    remove.addEventListener('click', () => removeDocument(document.id));
    item.append(glyph, copy, remove);
    documentList.append(item);
  }
}

async function removeDocument(id) {
  try {
    await api(`/api/documents/${encodeURIComponent(id)}`, { method: 'DELETE' });
    await refreshDocuments();
    await refreshHealth();
  } catch (error) {
    setUploadStatus(error.message, true);
  }
}

async function uploadFiles(fileList) {
  const files = [...fileList];
  if (!files.length) return;
  for (const file of files) {
    setUploadStatus(`Indexing ${file.name}…`);
    const body = new FormData();
    body.append('file', file);
    try {
      const result = await api('/api/documents', { method: 'POST', body });
      setUploadStatus(`${result.filename} indexed · ${result.chunk_count} chunks`);
    } catch (error) {
      setUploadStatus(`${file.name}: ${error.message}`, true);
    }
  }
  fileInput.value = '';
  await refreshDocuments();
  await refreshHealth();
}

async function refreshHealth() {
  try {
    const state = await api('/api/health');
    statusBox.classList.toggle('online', state.ready);
    statusBox.classList.toggle('offline', state.ollama !== 'online');
    if (state.ready) {
      statusLabel.textContent = `Ready · ${state.chat_model}`;
    } else if (state.ollama !== 'online') {
      statusLabel.textContent = 'Ollama is offline';
    } else {
      const missing = [
        !state.chat_model_ready && state.chat_model,
        !state.embedding_model_ready && state.embedding_model,
      ].filter(Boolean);
      statusLabel.textContent = `Pull model${missing.length > 1 ? 's' : ''} · ${missing.join(', ')}`;
    }
  } catch {
    statusBox.classList.add('offline');
    statusBox.classList.remove('online');
    statusLabel.textContent = 'App connection unavailable';
  }
}

function appendMessage(role, text, sources = []) {
  welcome.hidden = true;
  const message = document.createElement('article');
  message.className = `message ${role}`;
  const label = document.createElement('div');
  label.className = 'message-label';
  label.textContent = role === 'user' ? 'You' : 'Local Knowledge';
  const body = document.createElement('div');
  body.className = 'message-body';
  body.textContent = text;
  message.append(label, body);

  if (sources.length) {
    const details = document.createElement('details');
    details.className = 'sources';
    const summary = document.createElement('summary');
    summary.textContent = `Retrieved evidence · ${sources.length} ${sources.length === 1 ? 'chunk' : 'chunks'}`;
    details.append(summary);
    for (const [index, source] of sources.entries()) {
      const item = document.createElement('div');
      item.className = 'source-item';
      const heading = document.createElement('div');
      heading.className = 'source-heading';
      const origin = document.createElement('span');
      origin.textContent = `Source ${index + 1} · ${source.filename}${source.page ? ` · page ${source.page}` : ''}`;
      const score = document.createElement('span');
      score.className = 'source-score';
      score.textContent = `cosine ${source.score.toFixed(3)}`;
      const excerpt = document.createElement('p');
      excerpt.className = 'source-text';
      excerpt.textContent = source.text;
      heading.append(origin, score);
      item.append(heading, excerpt);
      details.append(item);
    }
    message.append(details);
  }

  messageList.append(message);
  message.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  appendMessage('user', question);
  questionInput.value = '';
  sendButton.disabled = true;
  const thinking = document.createElement('div');
  thinking.className = 'thinking';
  thinking.textContent = 'Searching your documents and asking the local model';
  messageList.append(thinking);
  thinking.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  try {
    const result = await api('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    thinking.remove();
    appendMessage('assistant', result.answer, result.sources);
  } catch (error) {
    thinking.remove();
    appendMessage('assistant', error.message);
  } finally {
    sendButton.disabled = false;
    questionInput.focus();
  }
});

document.querySelector('#choose-files').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => uploadFiles(fileInput.files));
uploadForm.addEventListener('submit', (event) => event.preventDefault());
refreshButton.addEventListener('click', refreshHealth);

for (const eventName of ['dragenter', 'dragover']) {
  uploadForm.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadForm.classList.add('dragging');
  });
}
for (const eventName of ['dragleave', 'drop']) {
  uploadForm.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadForm.classList.remove('dragging');
  });
}
uploadForm.addEventListener('drop', (event) => uploadFiles(event.dataTransfer.files));

questionInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    chatForm.requestSubmit();
  }
});

refreshDocuments().catch((error) => setUploadStatus(error.message, true));
refreshHealth();