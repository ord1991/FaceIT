function closeModal() {
    document.getElementById('overlay').style.display = 'none';
    document.getElementById('registration-modal').style.display = 'none';
    document.getElementById('registration-form').reset();
}

// Function to handle modal opening (placeholders for other JS logic to fill)
function openModal(faceId, imageUrl) {
    document.getElementById('reg-face-id').value = faceId;
    document.getElementById('reg-preview').src = imageUrl;
    document.getElementById('overlay').style.display = 'block';
    document.getElementById('registration-modal').style.display = 'block';
}
