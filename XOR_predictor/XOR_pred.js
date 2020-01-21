let model;

let training_data = [
    {
        inputs: tf.tensor2d([[0, 0]]),
        output:tf.tensor2d([[0]])
    },
    {
        inputs: tf.tensor2d([[0, 1]]),
        output:tf.tensor2d([[1]])
    },
    {
        inputs: tf.tensor2d([[1, 0]]),
        output:tf.tensor2d([[1]])
    },
    {
        inputs: tf.tensor2d([[1, 1]]),
        output:tf.tensor2d([[0]])
    }
];

function setup(){
    createCanvas(400, 400);
    background(0);
    model = tf.sequential();
    model.add(tf.layers.dense({
        units: 4,
        inputShape: [2],
        activation: 'sigmoid'
    }));
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));
    let sgdOpt = tf.train.sgd(0.5);
    model.compile({loss: tf.losses.meanSquaredError, optimizer: sgdOpt});

}

function letsTrain(){
    trainData().then(()=>{
        alert('Done all training!');
    });
}

async function trainData(){
    for(i = 0 ; i < 1000 ; i++){
        console.log(i);
        var r = Math.floor(Math.random() * 4);
        const response = await model.fit(training_data[r].inputs, training_data[r].output, {
            epochs: 10,
            shuffle: true
        });
    }
}

function letsVisualise(){
    let res = 10;
    rows = height / res;
    cols = width / res;
    for(i = 0 ; i < cols ; i++){
        for(j = 0; j < rows ; j++){
            let x = i / cols;
            let y = j / rows;  
            let inp = tf.tensor2d([[x , y]]);
            let opts = model.predict(inp).dataSync()[0];
            noStroke();
            fill(opts * 255);
            rect(i * res, j * res, res, res);
        }
    }
}


function draw (){}