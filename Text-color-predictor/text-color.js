let r, g, b;
let inputs, outputs;
let finColor
let model;

function getColor(){
    r = floor(random(255));
    g = floor(random(255));
    b = floor(random(255));
    console.log(r + g + b);
    background(r, g, b);
    inputs = tf.tensor2d([[r/255, g/255, b/255]]);
}


async function modelTrain(){
    for(i = 0; i< 100; i++){
        r = floor(random(255));
        g = floor(random(255));
        b = floor(random(255));
        background(r, g, b);
    
        inputs = tf.tensor2d([[r/255, g/255, b/255]]);
        if(r + g + b < 380){
            finColor = "white";
            outputs = tf.tensor2d([[1]]);
        }else{
            finColor = "black";
            outputs = tf.tensor2d([[0]]);
        }
        console.log(finColor);
        const response = await model.fit(inputs, outputs, {
            epochs: 10
        });
        console.log("Done fitting...");
        inputs.dispose();
        outputs.dispose();
    }
}

function setup() {
    createCanvas(600, 400);
    model = tf.sequential();
    hiddenLayer = tf.layers.dense({
        units: 4,
        activation: 'sigmoid',
        inputShape: [3]
    });
    model.add(hiddenLayer);
    outputLayer = tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
    });
    model.add(outputLayer);
    let sgdOpt = tf.train.sgd(0.5);

    model.compile({
        optimizer: sgdOpt,
        loss: tf.losses.meanSquaredError
    });
    textSize(40);
    getColor();
}

function letsTrain(){
    modelTrain().then(()=>{
        alert('Done all training!');
        getColor();
    });
}

function letsPredict(){
    const predVals = model.predict(inputs);
    let finVal = predVals.dataSync();
    if(finVal <= 0.5){
        alert("Black");
    }else{
        alert("White");
    }
    getColor();
}

function draw() {
    fill(0);
    text("Black", 100, 150);
    fill(128);
    text("Or", 275, 150);
    fill(255);
    text("White", 400, 150);
}