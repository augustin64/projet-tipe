/*
Le code concernant l'entrée par l'utilisateur d'un nombre a été adapté depuis le code suivant: (License MIT)
https://github.com/maneprajakta/Digit_Recognition_Web_App/blob/master/js/main.js 

On été rajoutées les fonctions permettant de communiquer avec le serveur
*/
// Variables globales
var canvas;
var ctx;
var touchX;
var touchY;
var mouseX;
var mouseY;
var mouseDown = 0;

const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
const RES_STANDARD = "Résultats réseau de neurones standard";
const RES_CONV = "Résultats réseau de neurones convolutif";
var canvasSize;

function init() {
    if (vw <= 1024) {
        canvasSize = vw * 0.95;
    } else {
        canvasSize = 280;
    }

    canvas = document.getElementById('digit-canvas');
    ctx = canvas.getContext('2d');
    
    ctx.fillStyle = "black";
    // Créer une zone de dessin
    canvas.width = canvasSize;
    canvas.height = canvasSize;

    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if(ctx)
    {
        canvas.addEventListener('mousedown', s_mouseDown, false);          
        canvas.addEventListener('mousemove', s_mouseMove, false);          
        window.addEventListener('mouseup', s_mouseUp, false);
        document.getElementById('clear').addEventListener("click", clear);

        canvas.addEventListener('touchstart', s_mouseDown, false);
        canvas.addEventListener('touchmove', s_touchMove, false);
        canvas.addEventListener('touchend', s_mouseUp, false);
    }
    clear();
}


function draw(ctx, x, y, isDown) {
    // isDown représente le fait que le chemin ait déjà commencé ou non
    // On ne dessine donc ue si le "stylo" est déjà posé
    if (isDown) {
        ctx.beginPath();
        // On choisit les options suivantes:
        ctx.strokeStyle = "white";
        ctx.lineWidth = "12";
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.closePath();
        ctx.stroke();
    }
    lastX = x;
    lastY = y;
}

// Mouse moves
function s_mouseDown() {
    mouseDown = true;
    draw(ctx, mouseX, mouseY, false);
}

function s_mouseUp() {
    mouseDown = false;
}

function s_mouseMove(e) {
    // Si la souris bouge et est cliquée, on souhaite dessiner un trait
    getMousePos(e);
    if (mouseDown) {
        draw(ctx, mouseX, mouseY, true);
    }
}

function getMousePos(e) {
    e.preventDefault();
    e.stopPropagation();
    if (e.offsetX) {
        mouseX = e.offsetX;
        mouseY = e.offsetY;
    } else if (e.layerX) {
        mouseX = e.layerX;
        mouseY = e.layerY;
    }
}

function s_touchMove(e) {
    var touch = e.touches[0];
    var mouseEvent = new MouseEvent("mousemove", {
      clientX: touch.clientX,
      clientY: touch.clientY
    });
    canvas.dispatchEvent(mouseEvent);
  }


function clear() {
    document.getElementById("result_fc").innerHTML = RES_STANDARD;
    document.getElementById("result_cnn").innerHTML = RES_CONV;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "black"; 
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}



function sendJSON(data,callback){
    // Creating a XHR object
    let xhr = new XMLHttpRequest();
    let url = "post";

    // open a connection
    xhr.open("POST", url, true);

    // Set the request header i.e. which type of content you are sending
    xhr.setRequestHeader("Content-Type", "application/json");

    // Create a state change callback
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {

            // Print received data from server
            callback(this.responseText);

        }
    };

    // Converting JSON data to string
    var data = JSON.stringify(data);

    // Sending data with the request
    xhr.send(data);
}


function addResults(elem, data, text) {
    elem.innerHTML = text;
    var elements = [];

    for (let i=0; i < data.length; i++) elements.push([data[i], i]);
    
    elements.sort(function(a, b) {
        a = a[1];
        b = b[1];
    
        return a < b ? -1 : (a > b ? 1 : 0);
    });

    let res = elements.sort().reverse();
    for (let j=0; j < res.length; j++) {
        elem.innerHTML += "<br/>"+res[j][1]+" : "+res[j][0];
    }
}


function getPrediction() {
    let imageSize = 28;
    let totalWidth = canvasSize;
    let totalHeight = canvasSize;
    let curWidth = 0;
    let curHeight = 0;
    let stepSize = totalWidth / imageSize;

    let tableau = [];
    let tmp_tableau = [];

    while (curHeight < totalHeight && tableau.length < imageSize) {
        curWidth = 0;
        tmp_tableau = [];
        while (curWidth < totalWidth  && tmp_tableau.length < imageSize) {
            data = ctx.getImageData(curWidth, curHeight, stepSize, stepSize);
            size = data.width * data.height;
            density = 0;

            for (let i=0;i < size;i++) {
                density += (data.data[i*4]+data.data[i*4+1]+data.data[i*4+2])/3;
            }
            density = density*1.0 / size;

            tmp_tableau.push(density);
            curWidth += stepSize;
        }    
        curHeight += stepSize;
        tableau.push(tmp_tableau);
    }

    return sendJSON({
        "type": "prediction",
        "dataset": "mnist",
        "data": tableau
    }, (data) => {
        data = JSON.parse(data);
        if (data["status"] != 200) {
            document.getElementById("result").innerHTML = "500 Internal Server Error";
        } else {
            addResults(document.getElementById("result_fc"), data["data"]["dense"], RES_STANDARD);
            addResults(document.getElementById("result_cnn"), data["data"]["cnn"], RES_CONV);
        }
    })
}