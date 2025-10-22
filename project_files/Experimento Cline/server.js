const express = require('express');
const path = require('path');

const app = express();
const port = 3034;

app.use(express.static(path.join(__dirname)));

app.listen(port, () => {
    console.log(`Server for Experimento Cline running on http://localhost:${port}`);
});
