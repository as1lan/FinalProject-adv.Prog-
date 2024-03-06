async function loadModel() {
    const model = await tf.loadLayersModel('model.json');
    return model;
}

async function predict() {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    const inputElement = document.getElementById('imageInput');

    if (inputElement.files.length === 0) {
        alert('Please select an image.');
        return;
    }

    const file = inputElement.files[0];

    const img = new Image();

    img.onload = async function () {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.drawImage(img, 0, 0, 48, 48);

        const imageData = ctx.getImageData(0, 0, 48, 48).data;
        const inputData = new Float32Array(48 * 48 * 1);
        for (let i = 0; i < imageData.length; i += 4) {
            inputData[i / 4] = imageData[i] / 255.0;
        }

        const input = tf.tensor4d(inputData, [1, 48, 48, 1]);

        const model = await loadModel();
        const [gender, ethnicity] = model.predict(input);

        const genderResult = gender.argMax(1).dataSync()[0] === 0 ? 'Male' : 'Female';
        const ethnicityResult = getEthnicityLabel(ethnicity.argMax(1).dataSync()[0]);

        document.getElementById('genderResult').innerText = genderResult;
        document.getElementById('ethnicityResult').innerText = ethnicityResult;
    };

    img.src = URL.createObjectURL(file);
}

function getEthnicityLabel(index) {
    const ethnicityLabels = ['White', 'Black', 'Indian', 'Asian', 'Hispanic'];
    return ethnicityLabels[index];
}
