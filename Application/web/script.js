async function openFileSelector() {
  var graphContainer = document.getElementById("graph_2");

  var width = graphContainer.offsetWidth;
  var height = graphContainer.offsetHeight;

  const base64Graphs = await eel.get_file_from_user(width, height)();
  if (base64Graphs) {
    displayGraph2(base64Graphs[0]);
    displayGraph3(base64Graphs[1]);
    displayGraph4(base64Graphs[2]);
    displayGraph5(base64Graphs[3]);
    sessionStorage.setItem("selectedFilePath", base64Graphs[4]);
  }
}
function displayGraph2(src) {
  document.getElementById("graph2").src = src;
}
function displayGraph3(src) {
  document.getElementById("graph3").src = src;
}
function displayGraph4(src) {
  document.getElementById("graph4").src = src;
}
function displayGraph5(src) {
  document.getElementById("graph5").src = src;
}
async function predict_Linear_Regressor() {
  var graphContainer = document.getElementById("graph_6");

  var width = graphContainer.offsetWidth;
  var height = graphContainer.offsetHeight;
  const savedFilePath = sessionStorage.getItem("selectedFilePath");
  const base64Graphs = await eel.predict_use_linear_regressor(savedFilePath)();
  if (base64Graphs) {
    displayGraph1(base64Graphs[0]);
    displayGraph6(base64Graphs[1]);
  }
}
async function predict_Decision_Tree_Regressor() {
  var graphContainer = document.getElementById("graph_6");

  var width = graphContainer.offsetWidth;
  var height = graphContainer.offsetHeight;
  const savedFilePath = sessionStorage.getItem("selectedFilePath");
  const base64Graphs = await eel.predict_use_decision_tree_regressor(
    savedFilePath
  )();
  if (base64Graphs) {
    displayGraph1(base64Graphs[0]);
    displayGraph6(base64Graphs[1]);
  }
}
async function predict_Random_Forest_Regressor() {
  var graphContainer = document.getElementById("graph_6");

  var width = graphContainer.offsetWidth;
  var height = graphContainer.offsetHeight;
  const savedFilePath = sessionStorage.getItem("selectedFilePath");
  const base64Graphs = await eel.predict_use_random_forest_regressor(
    savedFilePath
  )();
  if (base64Graphs) {
    displayGraph1(base64Graphs[0]);
    displayGraph6(base64Graphs[1]);
  }
}
function displayGraph1(src) {
  document.getElementById("graph1").src = src;
}
function displayGraph6(src) {
  document.getElementById("graph6").src = src;
}
