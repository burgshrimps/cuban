//add confirmation before refreshing
window.addEventListener("beforeunload", function (e) {
    var confirmationMessage = "\o/";

    (e || window.event).returnValue = confirmationMessage; //Gecko + IE
    return confirmationMessage;                            //Webkit, Safari, Chrome
});


//resize the image and draw it to the canvas
function resizeImage(imagePath, newWidth) {
    //create an image object from the path
    const originalImage = new Image();
    originalImage.src = imagePath;

    //get a reference to the canvas
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    document.getElementById("canvas").style.display = "flex"; 
    document.getElementById("positive").style.display = "block"; 
    document.getElementById("negative").style.display = "block"; 
    document.getElementById("back").style.display = "block";
    document.getElementById("stop").style.display = "block";

    //wait for the image to load
    originalImage.addEventListener('load', function () {

        //get the original image size and aspect ratio
        const originalWidth = originalImage.naturalWidth;
        const originalHeight = originalImage.naturalHeight;
        const aspectRatio = originalWidth / originalHeight;

        //if the new height wasn't specified, use the width and the original aspect ratio
        //calculate the new height
        newHeight = newWidth / aspectRatio;
        newHeight = Math.floor(newHeight);

        //set the canvas size
        canvas.width = newWidth;
        canvas.height = newHeight;

        //render the image
        ctx.drawImage(originalImage, 0, 0, newWidth, newHeight);
    });
}

//check if confirm or discard has been clicked and return true or false respectivley
function checkFlag(index) {
    var confirm = document.getElementById("positive");
    var discard = document.getElementById("negative");
    var back = document.getElementById("back");
    var stop = document.getElementById("stop");
    //returns promise to allow for async showing of variants figures
    return new Promise(acc => {
        //handles click event of the confirm button
        function handleClicktp() {
            acc('true');
        }
        //handles click event of the discard button
        function handleClickfp() {
            acc('false');
        }
        //handles click event of the back button
        function handleClickback() {
            acc('back');
        }
        //handles click event of the stop button
        function handleClickstop() {
            acc('stop');
        }
        //add click eventlistener to both buttons and optionally allow for key stroke as input
        function handleKeyInput(e) {
            if (e.key == 'ArrowRight') {
                acc('true');
            }
            if (e.key == 'ArrowLeft') {
                acc('false');
            }
        }
        document.addEventListener('keydown', handleKeyInput);
        confirm.addEventListener('click', handleClicktp);
        discard.addEventListener('click', handleClickfp); 
        back.addEventListener('click', handleClickback); 
        stop.addEventListener('click', handleClickstop)
    });
}

//async function to show individual variant plots
async function plot_variants(ids, sample, index=0, evaluations=[]) {
    var id = '';
    //iterate through all variant ids and return true or false after manual inspection
    for (var i = index, total=ids.length-1; i<=total; i++) {
        console.log(i)
        id = ids[i]
        console.log(ids[i])
        // if ((id.includes('INS')) || (id.includes('BND'))){
        //     image_src = `${sample}Insplot_Snapshots/${ids[i]}.png`;
        // }
        // else {
        //     image_src = `${sample}Samplot_Snapshots/${ids[i]}.png`;
        // }
        image_src = `${sample}/${ids[i]}.png`;

        resizeImage(image_src, 1000);
        check = await checkFlag(i)
        console.log(check)
        if (check == 'stop') {
            return evaluations;
        }
        else if (check == 'back'){
            evaluations.pop();
            evaluations = plot_variants(ids, sample, i-1, evaluations);
            break
        }
        else {
            evaluations.push(check);
        }
    }
    return evaluations;
}

//converts the csv or vcf data into a downloadable format
function download_file(filename, type, text) {
    filename = filename + type
    console.log(filename)
    var element = document.createElement('a');
    if (type=='tsv'){
        // Creating a Blob for having a vcf file format
        // and passing the data with type
        const blob = new Blob([text], { type: 'text/tsv' });
        // Creating an object for downloading url
        const url = window.URL.createObjectURL(blob)
        element.setAttribute('href', url);
    }
    else if (type=="vcf"){
        // Creating a Blob for having a vcf file format
        // and passing the data with type
        const blob = new Blob([text], { type: 'text/vcard' });
        // Creating an object for downloading url
        const url = window.URL.createObjectURL(blob)
        //var vcf_content = encodeURI('data:application/vcard;charset=utf-8,' + text)
        element.setAttribute('href', url);
    }
    element.setAttribute('download', filename);
    
    element.style.display = 'none';
    document.body.appendChild(element);

    element.click();

    document.body.removeChild(element);
}

//plot each variant interupted by manual inspection
async function evaluate_variants(lines, ids, type, sample, name, file_name) {
    const evaluations = await plot_variants(ids, sample)
    // if the evaluation has been iterrupted or return NAs for not yet inspected variants
    if (evaluations.length < (lines.length-1)){
        console.log(lines.length)
        evaluations.push(...Array((lines.length - 1) - evaluations.length).fill('NA'))
    };
    console.log(evaluations)
    console.log(evaluations.length)
    if (type=='tsv'){
    //create new CSV file by appending the evaluation to the previous lines
    new_file = [];
    for (var i = 0; i < lines.length-1; i++) {
        if (i == 0) {
            new_file.push(lines[i] + '\t' + `Confirmed (${name})`);
        }
        else {
            if (evaluations[i - 1]){
                new_file.push(lines[i] + '\t' + evaluations[i - 1]);
            }
        }
    }
    };
    if (type == 'vcf') {
        //create new VCF file by appending the evaluation to the previous lines
        new_file = [];
        for (var i = 0; i < lines.length - 1; i++) {
            if (lines[i].startsWith('##')) {
                new_file.push(lines[i]);
            }
            else if (lines[i].startsWith('#')) {
                new_file.push(`##INFO=<ID=CONFIRMED(${name}),Number=0,Type=Flag,Description="CONFIRMED IN MANUAL INSPECTION">`)
                new_file.push(lines[i]);
            }
            else {
                evaluation = evaluations.shift()
                console.log(evaluation)
                if (evaluation == 'true'){
                    line = lines[i].split('\t');
                    new_line = [line.slice(0, 7).join('\t'), line.slice(7) + `;CONFIRMED(${name})`,line.slice(8, ).join('\t')];
                    new_file.push(new_line.join('\t'));
                }
                else {
                    new_file.push(lines[i])
                }
            }
        }
    };
    final_output = new_file.join('\n')
    new_file_name = file_name.split('.').slice(0,-1).join('.') + '.' + name + '_curated.'
    console.log(new_file_name)
    // remove canvas and buttons
    document.getElementById("negative").style.display = 'none';
    document.getElementById("positive").style.display = 'none';
    document.getElementById("canvas").style.display = 'none';
    document.getElementById("back").style.display = 'none';
    document.getElementById("stop").style.display = 'none';

    // add download functionality and button
    document.getElementById("download").style.display = "flex";
    document.getElementById("download_button").addEventListener('click', function () { download_file(new_file_name, type, final_output)});

}

function readInput(evt) {
    evt.preventDefault();
    var formData = new FormData(evt.target);
    var formProps = Object.fromEntries(formData)
    console.log(formProps['name'])
    console.log(formProps['sample'])
    console.log(formProps['csv_file'])
    document.getElementById('input_form').style.display = 'none';
    var file = formProps['csv_file'];
    var sample = formProps['sample']
    var name = formProps['name']
    if (file) {
        var r = new FileReader();
        ids = [];
        if (file.type == "text/csv" || file.type == "text/tab-separated-values") {
            //load contents of the CSV file into an array of ids
            r.onload = function (e) {
                var contents = e.target.result;
                var lines = contents.split("\n");
                for (var i = 0; i < lines.length - 1; i++) {
                    var cells = lines[i].split("\t");
                    if (i > 0) {
                        ids.push(cells[0]);
                    }
                }
                evaluate_variants(lines, ids, 'tsv', sample, name, file.name)
            };
        };
        if (file.type == "text/vcard") {
            //load contents of the VCF file into an array of ids
            r.onload = function (e) {
                var contents = e.target.result;
                var lines = contents.split("\n");
                for (var i = 0; i < lines.length - 1; i++) {
                    if (lines[i].startsWith('#')){
                        continue;
                    };
                    var cells = lines[i].split("\t");
                    ids.push(cells[2]);
                }
                evaluate_variants(lines, ids, 'vcf', sample, name, file.name)
            };
        };
        r.readAsText(file);
    } else {
        alert("Please specify a CSV or VCF file!");
    }
}
