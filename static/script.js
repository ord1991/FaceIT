async function fetchUsers() {
    try {
        const response = await fetch('/users');
        const users = await response.json();
        const tbody = document.getElementById('user-list-body');
        tbody.innerHTML = '';
        if (users.length === 0) {
            tbody.innerHTML = '<tr><td colspan="3" style="text-align: center; padding: 20px; color: #7f8c8d;">No users registered</td></tr>';
            return;
        }
        users.forEach(user => {
            const tr = document.createElement('tr');
            tr.className = `status-${user.status}`;
            const isApproved = user.status === 'approved';
            const actionText = isApproved ? 'Blacklist' : 'Approve';
            const actionClass = isApproved ? 'btn-danger' : 'btn-success';
            const nextStatus = isApproved ? 'blacklisted' : 'approved';

            tr.innerHTML = `
                <td>${user.name}</td>
                <td>${user.status}</td>
                <td>
                    <button
                        class="btn btn-sm ${actionClass}"
                        onclick="updateStatus(${user.id}, '${nextStatus}')"
                        aria-label="${actionText} ${user.name}"
                    >
                        ${actionText}
                    </button>
                </td>
            `;
            tbody.appendChild(tr);
        });
    } catch (error) {
        console.error('Error fetching users:', error);
    }
}

async function fetchUnknownFaces() {
    try {
        const response = await fetch('/unknown_faces');
        const faces = await response.json();
        const container = document.getElementById('unknown-faces-container');
        container.innerHTML = '';
        faces.forEach(face => {
            const item = document.createElement('button');
            item.className = 'unknown-item';
            item.type = 'button';
            item.onclick = () => openModal(face.id, face.image_url);
            item.setAttribute('aria-label', 'Register this face');
            item.innerHTML = `<img src="${face.image_url}" alt="Unknown face">`;
            container.appendChild(item);
        });
    } catch (error) {
        console.error('Error fetching unknown faces:', error);
    }
}

async function updateStatus(userId, newStatus) {
    const formData = new FormData();
    formData.append('user_id', userId);
    formData.append('status', newStatus);

    try {
        const response = await fetch('/users/update_status', {
            method: 'POST',
            body: formData
        });
        if (response.ok) {
            fetchUsers();
        }
    } catch (error) {
        console.error('Error updating status:', error);
    }
}

function closeModal() {
    document.getElementById('overlay').style.display = 'none';
    document.getElementById('registration-modal').style.display = 'none';
    document.getElementById('registration-form').reset();
}

function openModal(faceId, imageUrl) {
    document.getElementById('reg-face-id').value = faceId;
    document.getElementById('reg-preview').src = imageUrl;
    document.getElementById('overlay').style.display = 'block';
    document.getElementById('registration-modal').style.display = 'block';
}

// Form submissions
document.getElementById('upload-form').onsubmit = async (e) => {
    e.preventDefault();
    const btn = e.target.querySelector('button');
    const originalText = btn.innerText;
    btn.disabled = true;
    btn.innerText = 'Uploading...';

    const formData = new FormData();
    formData.append('name', document.getElementById('upload-name').value);
    formData.append('status', document.getElementById('upload-status').value);
    formData.append('file', document.getElementById('upload-file').files[0]);

    try {
        const response = await fetch('/users/add', {
            method: 'POST',
            body: formData
        });
        if (response.ok) {
            document.getElementById('upload-form').reset();
            fetchUsers();
            alert('User registered successfully');
        } else {
            const err = await response.json();
            alert('Error: ' + err.detail);
        }
    } catch (error) {
        console.error('Error adding user:', error);
    } finally {
        btn.disabled = false;
        btn.innerText = originalText;
    }
};

document.getElementById('registration-form').onsubmit = async (e) => {
    e.preventDefault();
    const btn = e.target.querySelector('button[type="submit"]');
    const originalText = btn.innerText;
    btn.disabled = true;
    btn.innerText = 'Saving...';

    const formData = new FormData();
    formData.append('name', document.getElementById('reg-name').value);
    formData.append('status', document.getElementById('reg-status').value);
    formData.append('face_id', document.getElementById('reg-face-id').value);

    try {
        const response = await fetch('/users/add', {
            method: 'POST',
            body: formData
        });
        if (response.ok) {
            closeModal();
            fetchUsers();
            fetchUnknownFaces();
        } else {
            const err = await response.json();
            if (err.detail === "No face source provided") {
                alert('Error: The face detection has expired or is no longer available. Please try clicking a recent detection again.');
            } else {
                alert('Error: ' + err.detail);
            }
        }
    } catch (error) {
        console.error('Error adding user from tag:', error);
    } finally {
        btn.disabled = false;
        btn.innerText = originalText;
    }
};

// Initial fetch and polling
fetchUsers();
fetchUnknownFaces();
setInterval(fetchUsers, 5000);
setInterval(fetchUnknownFaces, 2000);
