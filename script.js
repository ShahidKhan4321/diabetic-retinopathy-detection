async function predict() {
    const formData = new FormData();
    const fileInput = document.getElementById('file');
    
    if (!fileInput.files[0]) {
        document.getElementById('prediction-result').innerText = "Please select an image first.";
        return;
    }

    formData.append("file", fileInput.files[0]);

    // Show the loading spinner
    document.getElementById('loading-spinner').style.display = 'block';
    document.getElementById('prediction-result').innerText = '';
    document.getElementById('prediction-result').classList.remove('success', 'failure'); // Reset any previous styles

    try {
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        // Hide the loading spinner after prediction
        document.getElementById('loading-spinner').style.display = 'none';

        if (data.prediction) {
            document.getElementById('prediction-result').innerText = "Prediction: " + data.prediction;
            document.getElementById('prediction-result').classList.add('success'); // Add success styling
        } else {
            document.getElementById('prediction-result').innerText = "Prediction failed. Please try again.";
            document.getElementById('prediction-result').classList.add('failure'); // Add failure styling
        }
    } catch (error) {
        // Hide the loading spinner on error
        document.getElementById('loading-spinner').style.display = 'none';
        console.error("Error:", error);
        document.getElementById('prediction-result').innerText = "Error occurred. Please try again later.";
        document.getElementById('prediction-result').classList.add('failure'); // Add failure styling
    }
}

function previewImage() {
    const fileInput = document.getElementById('file');
    const file = fileInput.files[0];
    
    if (file) {
        const reader = new FileReader();

        reader.onload = function(e) {
            const imagePreview = document.getElementById('image-preview');
            imagePreview.src = e.target.result;
            document.getElementById('image-preview-container').style.display = 'block';
        };

        reader.readAsDataURL(file);
    }
}
