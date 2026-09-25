/*
Entities in Titanic-Dataset.csv file from left to right

PassengerId | Survived | Pclass | Name | Sex | Age | SibSp | Parch | Ticket | Fare | Cabin | Embarked
*/

using Microsoft.ML;
using Microsoft.ML.Data;

/*
Creates an MLContext, which is the starting point for working with ML.NET.
It is used to load data, build the machine learning pipeline,
train the model, and evaluate how well the model performs.
*/
MLContext mlContext = new MLContext();

/*
Sets the path to the CSV file that contains the Titanic dataset.
The dataset contains information about the passengers,
including whether they survived or not.
*/
string csvPath = "Titanic-Dataset.csv";

/*
Reads the CSV file and converts each row into a TitanicData object.
hasHeader: true means that the first row contains the column names.
separatorChar: ',' means that the values are separated by commas.
allowQuoting: true allows text inside quotation marks to be handled correctly.
*/
IDataView data = mlContext.Data.LoadFromTextFile<TitanicData>(
    csvPath,
    hasHeader: true,
    separatorChar: ',',
    allowQuoting: true
);

/*
Splits the dataset into two parts: training data and test data.
80% is used to train the model.
20% is used to test the model and evaluate how accurate its predictions are.
*/
DataOperationsCatalog.TrainTestData splitData =
mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

/*
Creates a pipeline to prepare the data before training the model.
A pipeline is a series of steps that process the data in a specific order.
*/
var pipeline = mlContext.Transforms.Conversion.ConvertType(
    outputColumnName: "Label",
    inputColumnName: "Survived",
    outputKind: DataKind.Boolean)
/*
Converts the Sex values into numbers so the ML model can understand them.
*/
.Append(mlContext.Transforms.Categorical.OneHotEncoding(
    outputColumnName: "SexEncoded",
    inputColumnName: "Sex"))
/*
Combines the variables that the model uses to make predictions.
Age = the passenger's age.
Pclass = the passenger's class.
SexEncoded = the numerical values representing male and female.
*/
.Append(mlContext.Transforms.Concatenate(
    "Features",
    "Age",
    "Pclass",
    "SexEncoded"));

/*
Adds the machine learning algorithm to the pipeline.
SDCA Logistic Regression is used for binary classification,
where the model predicts one of two possible outcomes:
survived or did not survive.
*/
var trainingPipeLine = pipeline
.Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
    labelColumnName: "Label",
    featureColumnName: "Features"));

/*
Trains the model using 80% of the data from the CSV file.
The model learns the relationship between Age, Sex, Pclass, and Survived
so it can use these variables to make predictions.
*/
var model = trainingPipeLine.Fit(splitData.TrainSet);

/*
Uses the trained model to make predictions on the test data,
which was not used during training.
This allows us to measure how well the model performs on unseen data.
*/
IDataView predictions = model.Transform(splitData.TestSet);

/*
Evaluates the model's performance.
It uses Accuracy and F1 Score to help us understand
how well the model classifies the passengers.

Accuracy = the percentage of all predictions that are correct.

F1 Score = combines precision and recall into one measurement.
A higher F1 Score means the model has a better balance
between precision and recall.
*/
CalibratedBinaryClassificationMetrics metrics =
mlContext.BinaryClassification.Evaluate(
    predictions,
    labelColumnName: "Label");

Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");

/*
Creates a PredictionEngine that allows us to make predictions
for one passenger at a time using the trained model.
It is used later in the code to predict whether a passenger survived.
*/
var predictionEngine = mlContext.Model.CreatePredictionEngine<TitanicPredictionInput, TitanicPrediction>(model);

Console.WriteLine();
Console.WriteLine("🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊");
Console.WriteLine();
Console.WriteLine(" ⭕⭕⭕ Titanic Survivor Prediction ⭕⭕⭕");
Console.WriteLine();
Console.WriteLine("🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊🌊");
Console.WriteLine();

/*
Age must be between 0 and 100.
*/
Console.Write("Enter age: ");
float age;

while (!float.TryParse(Console.ReadLine(), out age) || age < 0 || age > 100)
{
    Console.WriteLine("Enter a valid age between 0 and 100.");
    Console.Write("Enter age: ");
}

/*
Sex must be either male or female.
*/
Console.Write("Enter sex (male/female): ");
string sex = Console.ReadLine()!.ToLower();

while (sex != "male" && sex != "female")
{
    Console.WriteLine("Enter a valid sex: male or female.");
    Console.Write("Enter sex (male/female): ");
    sex = Console.ReadLine()!.ToLower();
}

/*
pclass must be either 1, 2 or 3.
*/
Console.Write("Enter passenger class (1/2/3): ");
float pclass;

while (!float.TryParse(Console.ReadLine(), out pclass) ||
(pclass != 1 && pclass != 2 && pclass != 3))
{
    Console.WriteLine("Enter a valid passenger class.");
    Console.Write("Enter passenger class (1/2/3): ");
}

/*
Creates an object that represents a new passenger.
The passenger's information is passed to the trained ML model
to make a prediction.
*/
TitanicPredictionInput input = new TitanicPredictionInput
{
    Age = age,
    Sex = sex,
    Pclass = pclass
};

/*
Sends the passenger's information to the trained model.
The model uses this information to predict whether the passenger survived.
*/
TitanicPrediction prediction = predictionEngine.Predict(input);

/*

*/
if (prediction.Survived)
{
    Console.WriteLine("Prediction: The passenger survived.");
}
else
{
    Console.WriteLine("Prediction: The passenger did not survive.");
}
// Prints the models estimated probability
Console.WriteLine($"Probability: {prediction.Probability:P2}");

Console.WriteLine("Press Enter to exit...");
Console.ReadLine();


/*
Defines how each row in the CSV file should be read.
LoadColumn specifies which CSV column each property belongs to.
This information is used when the dataset is loaded.
*/
public class TitanicData
{
    [LoadColumn(1)] public float Survived { get; set; }
    [LoadColumn(2)] public float Pclass { get; set; }
    [LoadColumn(4)] public string Sex { get; set; } = "";
    [LoadColumn(5)] public float Age { get; set; }
};

/*
Defines the input data for a passenger entered by the user.
This information is passed to the trained model to make a prediction.
*/
public class TitanicPredictionInput
{
    public float Survived { get; set; }

    public float Pclass { get; set; }

    public string Sex { get; set; } = "";

    public float Age { get; set; }
}

/*
Contains the result from the ML model.
Survived shows the model's prediction.
Probability shows how confident the model is in its prediction.
*/
public class TitanicPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Survived { get; set; }

    public float Probability { get; set; }
}

