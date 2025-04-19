// Elementy terminala
const terminalInput = document.getElementById('terminal-input');
const terminalOutput = document.getElementById('terminal-output');

let inputData = {};

const inputFields = [
    'Series date', 'Length Overall', 'Maximum Beam', 'Draft', 'Displacement', 'DLR', 'IMS Division', 'Dynamic Allowance', 'Age Allowance', 'Mainsail measured', 'Mainsail rated', 'Headsail Luffed measured', 'Headsail Luffed rated', 'Symmetric measured', 'Symmetric rated', 'Mizzen measured', 'Mizzen rated', 'Headsail Flying measured', 'Headsail Flying rated', 'Asymmetric measured', 'Asymmetric rated', 'Quad. Mainsail measured', 'Quad. Mainsail rated', 'Mizzen Staysail measured', 'Mizzen Staysail rated'
];

let currentFieldIndex = 0;

// function for displaying prompt 
function displayPrompt() {
    terminalOutput.innerHTML = `Please enter "${inputFields[currentFieldIndex]}":<br>`;
}

displayPrompt();

terminalInput.addEventListener('keydown', function(event) {
    if (event.key === 'Enter') {
        const value = terminalInput.value.trim();
        if (value === '') {
            terminalOutput.innerHTML = `Input cannot be empty. Please enter "${inputFields[currentFieldIndex]}":<br>`;
        } 
        else {
            inputData[inputFields[currentFieldIndex]] = value;
            terminalInput.value = '';
            currentFieldIndex++;
            if (currentFieldIndex < inputFields.length) {
                displayPrompt();
            } else {
                terminalOutput.innerHTML = `All inputs received. Processing data...<br>`;
                terminalInput.disabled = true; 
                submitData(inputData);
            }
        }
    }
});

// Funkcja wysyłająca dane do FastAPI
function submitData(data) {
    fetch('http://localhost:8000/predict_input', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ "input_data": data })
    })
    .then(response => response.json())
    .then(result => {
        terminalOutput.innerHTML = `Data processed successfully.<br>`;
        updateTable(result);
        // Wyświetlamy przycisk resetu
        document.getElementById('reset-button').style.display = 'inline-block';
    })
    .catch(error => {
        console.error('Error:', error);
        terminalOutput.innerHTML = `An error occurred while processing data.<br>`;
    });
}

// Funkcja aktualizująca tabelę wyników
function updateTable(data) {
    const table = document.getElementById('result-table').getElementsByTagName('tbody')[0];
    const rows = table.rows;

    const rowLabels = ['Beat Angles', 'Beat VMG', '52°', '60°', '75°', '90°', '110°', '120°', '135°', '150°', 'Run VMG', 'Gybe Angles'];
    const windVelocities = ['6 kt', '8 kt', '10 kt', '12 kt', '14 kt', '16 kt', '20 kt', '24 kt'];

    for (let i = 0; i < rows.length; i++) {
        const rowLabel = rowLabels[i];
        const cells = rows[i].cells;
        for (let j = 1; j < cells.length; j++) {
            const windVelocity = windVelocities[j - 1];
            const key = `${rowLabel} ${windVelocity}`;
            cells[j].innerText = data[key] !== undefined ? data[key] : '-';
        }
    }
}

// Funkcja resetująca formularz i tabelę
document.getElementById('reset-button').addEventListener('click', function() {
    // Resetujemy zmienne
    inputData = {};
    currentFieldIndex = 0;
    terminalInput.value = '';
    terminalInput.disabled = false;
    // Czyścimy tabelę
    clearTable();
    // Ukrywamy przycisk resetu
    this.style.display = 'none';
    // Wyświetlamy początkowy prompt
    displayPrompt();
});

// Funkcja czyszcząca tabelę wyników
function clearTable() {
    const table = document.getElementById('result-table').getElementsByTagName('tbody')[0];
    const rows = table.rows;

    for (let i = 0; i < rows.length; i++) {
        const cells = rows[i].cells;
        for (let j = 1; j < cells.length; j++) {
            cells[j].innerText = '';
        }
    }
}
