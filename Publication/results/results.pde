//I have a capstone project (which requires local files in .csv format) //<>//
//I am demonstrating that I can get data from a live source using load data function but doing nothign with it.
//I proceed to to fulful the curves assingment requirements with data from my capstone project. 

JSONObject json;
Table table;
int[] ySignalCoordinates;
float xTime;

void settings() {
  size(3840,2160);
}

void setup() {
  //background(0,81,162);
  background(255);
  fill(53,76,161);
  noStroke();
  //stroke(3);



 
  table = loadTable("seg_00a37e.csv", "header"); //Load data from my captsone project
  int recordCount = table.getRowCount();
  ySignalCoordinates = new int[recordCount];

  beginShape();
  for (int i=0;i<recordCount;i+=1,xTime+=.0256) {
    ySignalCoordinates[i] = table.getInt(i,"acoustic_data");
    curveVertex(xTime,(ySignalCoordinates[i]*8)+height/2);
  }
  endShape();
  save("samplevisual.jpg");
}
