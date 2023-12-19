document.getElementById('box').addEventListener('click', function() {
    document.getElementById('file').click();
});


function generateReferenceImages() {
    document.getElementById('step1').style.display = 'none';
    document.getElementById('loading').style.display = 'block';
    // Qui dovresti inserire la logica per interagire con il tuo server Flask e ottenere lo stato di avanzamento.
    // Una volta completato, puoi passare alla schermata successiva.
    // Per ora, come esempio, ho inserito un timeout per simulare un ritardo.
    setTimeout(function() {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('step2').style.display = 'block';
        document.querySelector('.progress').style.width = '66%';
    }, 2000); // Simula un ritardo di 2 secondi
}

function requestFinalImages() {
    document.getElementById('step2').style.display = 'none';
    document.getElementById('step3').style.display = 'block';
    document.querySelector('.progress').style.width = '100%';
}

        var isAdvancedUpload = function() {
            var div = document.createElement('div');
            return (('draggable' in div) || ('ondragstart' in div && 'ondrop' in div)) && 'FormData' in window && 'FileReader' in window;
        }();

    var $form = document.querySelector('.box');

        if (isAdvancedUpload) {
            $form.classList.add('has-advanced-upload');

            $form.addEventListener('drag dragstart dragend dragover dragenter dragleave drop', function(e) {
                e.preventDefault();
                e.stopPropagation();
            });
            $form.addEventListener('dragover dragenter', function() {
                $form.classList.add('is-dragover');
            });
            $form.addEventListener('dragleave dragend drop', function() {
                $form.classList.remove('is-dragover');
            });
            $form.addEventListener('drop', function(e) {
                var droppedFiles = e.dataTransfer.files;
            });
        }
