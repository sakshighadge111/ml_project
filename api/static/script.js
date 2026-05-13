// This file handles form submission and prediction API call

document.addEventListener("DOMContentLoaded", function () {

    const form = document.getElementById("prediction-form");

    form.addEventListener("submit", async function (e) {

        e.preventDefault();

        console.log("Predict button clicked");

        const data = {

            Administrative: parseInt(document.getElementById("Administrative").value),

            Administrative_Duration: parseFloat(document.getElementById("Administrative_Duration").value),

            Informational: parseInt(document.getElementById("Informational").value),

            Informational_Duration: parseFloat(document.getElementById("Informational_Duration").value),

            ProductRelated: parseInt(document.getElementById("ProductRelated").value),

            ProductRelated_Duration: parseFloat(document.getElementById("ProductRelated_Duration").value),

            BounceRates: parseFloat(document.getElementById("BounceRates").value),

            ExitRates: parseFloat(document.getElementById("ExitRates").value),

            PageValues: parseFloat(document.getElementById("PageValues").value),

            SpecialDay: parseFloat(document.getElementById("SpecialDay").value),

            Month: parseInt(document.getElementById("Month").value),

            OperatingSystems: parseInt(document.getElementById("OperatingSystems").value),

            Browser: parseInt(document.getElementById("Browser").value),

            Region: parseInt(document.getElementById("Region").value),

            TrafficType: parseInt(document.getElementById("TrafficType").value),

            VisitorType: parseInt(document.getElementById("VisitorType").value),

            Weekend: parseInt(document.getElementById("Weekend").value)

        };

        const resultDiv = document.getElementById("result");

        resultDiv.classList.remove("hidden");

        resultDiv.innerHTML = "Predicting...";

        try {

            const response = await fetch("/predict", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(data)

            });

            const result = await response.json();

            console.log(result);

            if (result.prediction === 1) {

                resultDiv.innerHTML =
                    `✅ ${result.message}<br>
                    Probability: ${(result.probability * 100).toFixed(2)}%`;

            } else {

                resultDiv.innerHTML =
                    `❌ ${result.message}<br>
                    Probability: ${(result.probability * 100).toFixed(2)}%`;
            }

        } catch (error) {

            console.error(error);

            resultDiv.innerHTML = "Error occurred while predicting.";

        }

    });

});