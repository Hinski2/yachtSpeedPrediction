// Terminal elements
const fileUploadInput = document.getElementById('file-upload');
const terminalOutput = document.getElementById('terminal-output');
const terminalInput = document.getElementById('terminal-input');

// Ensure no redundant click events are added
terminalInput.addEventListener('click', function() {
    fileUploadInput.click();
}, { once: true }); // This ensures the event listener is executed only once

// Event listener for file upload
fileUploadInput.addEventListener('change', function(event) {
    const file = event.target.files[0];

    if (!file) {
        terminalOutput.innerHTML += "<br>No file selected. Please try again.";
        return;
    }

    // Check file extension
    const validExtensions = ["xls", "xlsx"];
    const fileExtension = file.name.split('.').pop().toLowerCase();

    if (!validExtensions.includes(fileExtension)) {
        terminalOutput.innerHTML += `<br>Invalid file type: ${fileExtension}. Please upload a valid Excel file (.xls or .xlsx).`;
        return;
    }

    // File is valid - notify the user
    terminalOutput.innerHTML += `<br>File "${file.name}" is being processed...`;

    // Create FormData to send the file to the server
    const formData = new FormData();
    formData.append('file', file);

    // Send the file to the server using fetch
    fetch('http://localhost:8000/upload_excel', {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        // Create a link to download the processed file
        const downloadUrl = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = 'processed_output.xlsx';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        terminalOutput.innerHTML += `<br>File processed successfully. The download should start automatically.`;
    })
    .catch(error => {
        console.error('Error:', error);
        terminalOutput.innerHTML += `<br>An error occurred while processing the file. Please try again.`;
    });
});
