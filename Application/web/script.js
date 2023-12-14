async function openFileSelector() {
  var graphContainer = document.getElementById("graph_2");
  var width = graphContainer.offsetWidth;
  var height = graphContainer.offsetHeight;
  await eel.get_file_from_user(width, height)(displayGraph); // Call Python function
}
function displayGraph(src) {
  document.getElementById("graph2").src = src;
}
