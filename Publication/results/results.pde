
Table table;
float[] ySignalCoordinates;
float xTime;

void settings() {
  size(3840,2160);
}

void setup() {
  background(255);
  fill(53,76,161,25);
  //noFill();
  //noStroke();
  stroke(53,76,161);
  strokeWeight(3);
  strokeJoin(ROUND);

 
  table = loadTable("y_nolog.csv", "header"); //Load data from my captsone project
  int recordCount = table.getRowCount();
  println(recordCount);
  ySignalCoordinates = new float[recordCount];
  //yNextSignalCoordinates = new float[recordCount];

  beginShape();
  for (int i=0;i<recordCount;i+=1,xTime+=.9155) { //<>//
    ySignalCoordinates[i] = table.getFloat(i,"acoustic_data");
    
    //line(xTime, (ySignalCoordinates[i])+height/2, xTime+=.9155, ySignalCoordinates[i+1]+height/2);
    curveVertex(xTime,((ySignalCoordinates[i]*10)+height/2));
    //point(xTime,(ySignalCoordinates[i]*10)+height/2);
  }
  endShape();
}
