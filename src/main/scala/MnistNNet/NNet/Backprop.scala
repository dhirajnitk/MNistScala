package MnistNNet.NNet
import MnistNNet.MnistLoader
import jeigen._
import jeigen.Shortcuts._

import scala.util.Random._
import scala.math._
object CoreConf {
  var Lmbda: Vector[Float] = Vector.empty[Float]
  var LMomentum:Vector[Float] = Vector.empty[Float]
  var LRates:Vector[Float] = Vector.empty[Float]
  var LTransFunctions:Vector[TransferFunction] = Vector.empty[TransferFunction]
}
trait NNetConf {
   var weights: Vector[DenseMatrix] = Vector.empty[DenseMatrix]
   var biases: Vector[DenseMatrix] = Vector.empty[DenseMatrix]
   def Lmbda: Vector[Float] = CoreConf.Lmbda
   def Momentum: Vector[Float] = CoreConf.LMomentum
   def LRates: Vector[Float] = CoreConf.LRates
   def LTransFunctions:Vector[TransferFunction] = CoreConf.LTransFunctions
   def getGaussMatrix(a: Int, b: Int): Array[Array[Double]] = {
    val dataArr:  Array[Array[Double]] = Array.ofDim[Double](a , b)
    for {
      i <- 0 to a - 1
      j <- 0 to b - 1
    }
      {
        dataArr(i)(j) = nextGaussian()
      }
     dataArr
  }
}
/*
object  Regularilzation extends Enumeration {
  type Regularilzation = Value
  val UnRegularized, L2, L1 = Value
}
*/

class  initWtUpdate (batchSize : Int, layers: List[Int], newNNetConf: NNetConf, oldNNetConf: NNetConf) extends NNetConf{
  //protected var eta : Float
  //protected val batchSize : Int = 0
  //protected val layers: List[Int] = Nil
  //protected val newNNetConf: NNetConf = new NNetConf {}
  //protected val oldNNetConf: NNetConf = new NNetConf {}
  def zeroMatrix( temp : NNetConf) :Unit = {
    for (i <-1 to  layers.length - 1){
      temp.weights=  temp.weights :+ zeros(layers(i-1), layers(i))
      temp.biases=  temp.biases :+ zeros(1, layers(i))

    }
  }
  def largeInitialize(layers:List[Int]): Unit = {
    for (i <-1 to  layers.length - 1){

      val wtGaussMat = getGaussMatrix(layers(i-1), layers(i))
      newNNetConf.weights=  newNNetConf.weights :+ new DenseMatrix(wtGaussMat)
      oldNNetConf.weights=  oldNNetConf.weights :+ new DenseMatrix(wtGaussMat)
      val bsGaussMat = getGaussMatrix(1, layers(i))
      newNNetConf.biases=  newNNetConf.biases :+ new DenseMatrix(bsGaussMat)
      oldNNetConf.biases=  oldNNetConf.biases :+ new DenseMatrix(bsGaussMat)

    }
  }
  def defaultInitialize(layers:List[Int]):Unit = {
    for (i <-1 to  layers.length - 1){

      val wtGaussMat = getGaussMatrix(layers(i-1), layers(i))
      newNNetConf.weights=  newNNetConf.weights :+ new DenseMatrix(wtGaussMat).div(sqrt(layers(i-1)))
      oldNNetConf.weights=  oldNNetConf.weights :+ new DenseMatrix(wtGaussMat).div(sqrt(layers(i-1)))
      val bsGaussMat = getGaussMatrix(1, layers(i))
      newNNetConf.biases=  newNNetConf.biases :+ new DenseMatrix(bsGaussMat)
      oldNNetConf.biases=  oldNNetConf.biases :+ new DenseMatrix(bsGaussMat)

    }
  }
   def updateWtsBiases(temp: NNetConf) : Unit = {
    for (i <-0 to  temp.weights.length - 1) {
      val eta = LRates(i)
      val momentum = Momentum(i)
      val wts = (temp.weights(i).mul(eta/batchSize));
      val biases = (temp.biases(i).mul(eta/batchSize));
      //newNNetConf.weights = newNNetConf.weights.updated(i,newNNetConf.weights(i).sub(wts).add(oldNNetConf.weights(i).mul(momentum)))
      newNNetConf.weights = newNNetConf.weights.updated(i,newNNetConf.weights(i).sub(wts))
      newNNetConf.biases = newNNetConf.biases.updated(i,newNNetConf.biases(i).sub(biases))

      oldNNetConf.weights = oldNNetConf.weights.updated(i,wts)
      oldNNetConf.biases = oldNNetConf.biases.updated(i, biases)
    }
  }

  /*
  MatrixXf feedforward(MatrixXf data) {
			for (int i = 0; i <BWts.biases.size(); i++) {
				data = (data * BWts.weights[i]) + BWts.biases[i].replicate((int)data.rows(), 1);
				data = sigmoid(data);
			}
			return data;
		}
*/

}
/*
trait UnRegularizedWtUpdate extends  initWtUpdate {
   override def updateWtsBiases(temp: NNetConf) : Unit = {
    for (i <-0 to  temp.weights.length - 1) {
      val eta = lRates(i)
      oldNNetConf.weights.updated(i,newNNetConf.weights(i))
      oldNNetConf.biases.updated(i,newNNetConf.biases(i))
      val wts = newNNetConf.weights(i).sub(temp.weights(i).mul(eta/batchSize));
      newNNetConf.weights.updated(i,wts)
      val biases = newNNetConf.biases(i).sub(temp.weights(i).mul(eta/batchSize));
      newNNetConf.weights.updated(i,biases)
    }
  }
}*/

class RegularizedWtUpdate(batchSize : Int, layers: List[Int], newNNetConf: NNetConf, oldNNetConf: NNetConf
                         , noTrainSmpl : Int ) extends  initWtUpdate(batchSize, layers , newNNetConf , oldNNetConf ){
  //protected val lmbda : Float = 0.1F
  //protected val noTrainSmpl : Int = 50000
   override def updateWtsBiases(temp: NNetConf) : Unit = {
    for (i <-0 to  temp.weights.length - 1) {
      val eta = LRates(i)
      val lmbda = Lmbda(i)
      val momentum = Momentum(i)
	    val wts = (temp.weights(i).mul(eta/batchSize));
      val biases = (temp.biases(i).mul(eta/batchSize));
      //val wts = newNNetConf.weights(i).mul(1 - eta * lmbda/noTrainSmpl).sub(temp.weights(i).mul(eta/batchSize))
      //newNNetConf.weights = newNNetConf.weights.updated(i,wts.add(oldNNetConf.weights(i).mul(momentum)))
	    newNNetConf.weights = newNNetConf.weights.updated(i,newNNetConf.weights(i).mul(1 - eta * lmbda/noTrainSmpl).sub(wts).add(oldNNetConf.weights(i).mul(momentum)))
      newNNetConf.biases = newNNetConf.biases.updated(i,newNNetConf.biases(i).sub(biases))
	    oldNNetConf.weights = oldNNetConf.weights.updated(i,wts)
      oldNNetConf.biases = oldNNetConf.biases.updated(i, biases)
    }
  }
}

trait Regularilzation
case object UnRegularized extends  Regularilzation
case object L2 extends Regularilzation
case object L1 extends Regularilzation

trait WtSettings
case object Large extends WtSettings
case object Default extends WtSettings


class Network( layers: List[Int], noTrainSmpl : Int , batchSize : Int, lrates : Vector[Float], lmomentum: Vector[Float],
               ltransFunctions : Vector[TransferFunction], lcFunction : SLP, wtSetting: WtSettings,
               regSettings : Regularilzation, lmbda: Vector[Float]   )  {
  CoreConf.Lmbda = lmbda
  CoreConf.LRates = lrates
  CoreConf.LMomentum = lmomentum
  CoreConf.LTransFunctions = ltransFunctions
  protected val cFunction:SLP = lcFunction
  protected val newNNetConf: NNetConf = new NNetConf {}
  protected val oldNNetConf: NNetConf = new NNetConf {}
  protected var wtUpdate : initWtUpdate = new initWtUpdate (batchSize, layers , newNNetConf , oldNNetConf ){}
  def initConfigSettings():Unit={
    regSettings match  {
      case L2 => wtUpdate = new RegularizedWtUpdate (batchSize, layers , newNNetConf , oldNNetConf, noTrainSmpl )
      case _ => ;
    }
     wtSetting match {
       case Large => wtUpdate.largeInitialize(layers)
       case Default => wtUpdate.defaultInitialize(layers)
     }

  }
  def feedForward(data: DenseMatrix) : DenseMatrix = {
    var output:DenseMatrix = data
    for (i <-0 to  newNNetConf.weights.length - 1) {
      output = CoreConf.LTransFunctions(i).transfer(output.mmul(newNNetConf.weights(i)).add(ones(output.rows, 1).mmulj(newNNetConf.biases(i))))

    }
    output
  }
  protected def createLabelMatrix(l_in : DenseMatrix) : DenseMatrix = {
   val lVec: DenseMatrix = zeros(l_in.rows, 10)
    for (index <- 0 to l_in.rows - 1) {
      lVec.set(index, l_in.get(index, 0).toInt, 1.0)
    }
    lVec
  }
  protected def backprop(data:DenseMatrix, lvec:DenseMatrix, temp : NNetConf) : Unit= {
    var activations: Vector[DenseMatrix] = Vector.empty[DenseMatrix]
    var zs:Vector[DenseMatrix] =  Vector.empty[DenseMatrix]
    activations=  activations :+ data
    // Forward pass
    var outdata:DenseMatrix = data
    for (i <-0 to  newNNetConf.weights.length - 1) {
      val z= outdata.mmul(newNNetConf.weights(i)).add(ones(outdata.rows, 1).mmulj(newNNetConf.biases(i)))
      zs = zs :+ z
      outdata = newNNetConf.LTransFunctions(i).transfer(z)
      activations=  activations :+ outdata
    }
    // Backward pass
    var delta:DenseMatrix = cFunction.delta(zs(zs.length -1),activations(activations.length -1), lvec)
    temp.biases = temp.biases.updated(temp.biases.length - 1, delta.sumOverRows())
    temp.weights = temp.weights.updated(temp.weights.length - 1,activations(activations.length - 2).t().mmul(delta))
    for (layer <-2 to  layers.length - 1) {
      val z = zs(zs.length - layer)
      val tFun:TransferFunction = newNNetConf.LTransFunctions(newNNetConf.LTransFunctions.length - layer)
      if(tFun.isInstanceOf[SoftMaxTransferFunction])
        delta = tFun.asInstanceOf[SoftMaxTransferFunction].flatMult(delta.mmul(newNNetConf.weights(newNNetConf.biases.length - layer + 1).t()),
          tFun.transferredSlope(z))
      else
        delta = delta.mmul(newNNetConf.weights(newNNetConf.biases.length - layer + 1).t()).mul(tFun.transferredSlope(z))
      temp.biases = temp.biases.updated(temp.biases.length - layer, delta.sumOverRows())
      //println(temp.biases)
      temp.weights = temp.weights.updated(temp.weights.length - layer,activations(activations.length - layer - 1).t().mmul(delta))
    }
  }
  //protected val newNNetConf: NNetConf
  //protected val oldNNetConf: NNetConf
  //protected val eta : Float
  //protected val lmbda : Float
  //final def InitAllSettings()

  /*
  final def learn(trainData:DenseMatrix,trainLbl:DenseMatrix, epochs : Int, regSettings : Regularilzation ):Unit = {

  }
  */

  def sgdMiniBatch (data: DenseMatrix, l_in : DenseMatrix): Unit = {
    val temp: NNetConf = new NNetConf {}
    wtUpdate.zeroMatrix(temp)
    /*
    val lVec: DenseMatrix  = createLabelMatrix(l_in)
    backprop(data, lVec, temp)
    */
    backprop(data, l_in, temp)
    wtUpdate.updateWtsBiases(temp)
  }


}
trait SLP extends CostFunction with TransferFunction
// Possible combination of SLP supported in this network
/*
trait SigmoidQuadraticSLP extends SLP with QuadraticCost with SigmoidTransferFunction{
}
trait TanhQuadraticSLP extends SLP with QuadraticCost with TanhTransferFunction{
}
trait SoftMaxQuadraticSLP extends SLP with QuadraticCost with SoftMaxTransferFunction{
}

trait SigmoidCrossEntropySLP extends SLP with SigmoidCrossEntropyCost with SigmoidTransferFunction{
}
trait TanhCrossEntropySLP extends SLP with TanhCrossEntropyCost with TanhTransferFunction

trait SoftMaxLogLikelihoodSLP extends SLP with SoftMaxTransferLogLHoodCost with SoftMaxTransferFunction

trait SigmoidLogLikelihoodSLP extends SLP with SigmoidTransferLogLHoodCost with SigmoidTransferFunction
*/
object GenericMLP {
  /*
  val a= new SoftMaxTransferFunction  {}
  def calc () :DenseMatrix = {
    a.transfer(ones(2, 2))
  }

 var a:SLP = new SLP with  QuadraticCost with SigmoidTransferFunction {}
 var b:TransferFunction = new SigmoidTransferFunction {}
 */
  /*Initialized parameters of object*/
  def indexOfLargest(array: Array[Double]): Int = {
    val result = array.foldLeft(-1,Int.MinValue.toDouble,0) {
      case ((maxIndex, maxValue, currentIndex), currentValue) =>
        if(currentValue > maxValue) (currentIndex,currentValue,currentIndex+1)
        else (maxIndex,maxValue,currentIndex+1)
    }
    result._1
  }

  def scale(mat:DenseMatrix) = {
    val minX = mat.minOverCols().mmulj(ones(1,mat.cols))
    val maxX = mat.maxOverCols().mmulj(ones(1, mat.cols))
    mat.sub(minX).mul(2).div(maxX.sub(minX)).sub(1).get2dValues()
  }
def run() {

    var (testLbl, tLbl, testImg, tImg) = MnistLoader.getMnistImageData("C:\\Users\\dhira\\OneDrive\\MachineLearningCode\\NeuralNetwork\\Scala\\data")
    var trainLbl = tLbl.slice(0,50000)
    var trainImg = tImg.slice(0,50000)
    var validationImg = tImg.slice(50000,tImg.length)
    var validationLbl = tLbl.slice(50000, tImg.length)

    var testData = new DenseMatrix(testImg)
    val epochs = 30
    val layers: List[Int] = List(784, 30, 10)
    val noTrainSmp = 50000
    val batchSize = 10
    val lrates: Vector[Float] = Vector(1.0F, 2.0F)
    //Higher momemntum needs lower learning rates
    val lmomentum: Vector[Float] = Vector(0.0F, 0.0F)
    val ltransFunctions: Vector[TransferFunction] = Vector(new SoftPlusTransferFunction  {}   , new SoftMaxTransferFunction  {})
    val lcFunction: SLP = new SLP with SoftMaxQuadraticCost with SoftMaxTransferFunction
    val wtSettings: WtSettings = Default
    val regSettgins: Regularilzation = UnRegularized
    val lmbda: Vector[Float] = Vector(5.0F, 5.0F)
    val Net: Network = new Network(layers, noTrainSmp, batchSize, lrates, lmomentum, ltransFunctions, lcFunction, wtSettings, regSettgins, lmbda)
    /*
  Network( val layers: List[Int], val noTrainSmpl : Int , val batchSize : Int, val lrates : Vector[Float],
  val lmomentum: Vector[Float], val ltransFunctions : Vector[TransferFunction], val lcFunction : SLP,
   val wtSetting: WtSettings, val regSettings : Regularilzation, val lmbda: Vector[Float]   )
   */
    Net.initConfigSettings()
   //normalize data for tanh
   /*
   if(lcFunction.isInstanceOf[TanhTransferFunction]) {
     trainImg = scale(new DenseMatrix(trainImg))
     testImg = scale(new DenseMatrix(testImg))
   }
   */

   val startTime = System.currentTimeMillis();
    for ( ep <- 1 to epochs ) {
      val trainData = MnistLoader.shuffle(trainImg, trainLbl)
      trainImg = trainData._1
      trainLbl = trainData._2
      for ( j <- 0 to noTrainSmp-1 by batchSize){
        if(j + batchSize < noTrainSmp){
          Net.sgdMiniBatch(new DenseMatrix(trainImg.slice(j, j + batchSize)),new DenseMatrix(trainLbl.slice(j, j + batchSize)));
          }
          else{
          val minBatch = noTrainSmp - j;
          Net.sgdMiniBatch(new DenseMatrix(trainImg.slice(j, j + minBatch)),new DenseMatrix(trainLbl.slice(j, j + minBatch)));
          }
      }
      val output = Net.feedForward(testData)
      val countArr:Array[Double] = output.get2dValues().map( obj => {indexOfLargest(obj).toDouble
      })
      for ( row <- 0 to countArr.length - 1) {
        countArr(row) = countArr(row)-testLbl(row)(0)
        //println(countArr(row))
        //countArr.update(row, countArr(row)-testLbl(0)(row))
      }
      //println(countArr)
      val acc  = countArr.filter( tgt => tgt ==0.toDouble).length
      printf("Epoch:%d,  %d / %d\n",ep, acc,testLbl.length);
    }
     printf( "Total Time %d sec\n",(System.currentTimeMillis() - startTime)/1000);
  }

}
