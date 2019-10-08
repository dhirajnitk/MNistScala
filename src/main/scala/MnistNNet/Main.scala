package MnistNNet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import java.util.Random
import java.util.Arrays
/**
  * Created by dhira on 26-06-2017.
  */

import jeigen.Shortcuts._
import jeigen._
import org.nd4j.linalg.api.ndarray.INDArray

import sys.process._
import java.net.URL
import java.io.File


object Main {
  def fileDownloader(url: String, filename: String) = {
    new URL(url) #> new File(filename) !!
  }

  def average1(list: List[Double]): Double = list match {
    case head :: tail => tail.foldLeft((head, 1.0))((r, c) =>
      ((r._1 + (c / r._2)) * r._2 / (r._2 + 1), r._2 + 1))._1
    case Nil => Double.NaN
  }

  def average(list: List[Double]): Double = list match {
    case head :: tail => tail.foldLeft((head, 1.0))((r, c) =>
      ((r._1 * r._2 + c) / (r._2 + 1), r._2 + 1))._1

    case Nil => Double.NaN

  }

  def zeros(rows :Int , cols:Int):Array[Array[Double]]= {
    val result = Array.ofDim[Double](rows,cols)
    result
  }
  def rand(rows :Int , cols:Int):Array[Array[Double]]= {
    val result = Array.ofDim[Double](rows,cols)
    val  random = new Random();
    for (row <- 0 to rows - 1) {
      for (col <- 0 to cols - 1) {
        result(row)(col) = random.nextDouble()
      }
    }
    result
  }

  def byteFill(array:Array[Double],value:Double) : Array[Double] = {
    val len = array.length
    if(len>0)
      array(0) = value
    for (i <- 1 to len - 1) {
      System.arraycopy(array, 0, array, i, if (len - i < i)  (len - i) else i);
    }
    array
  }
  def byteFill(array:Array[Array[Double]],value:Double) = {
    val len = array(0).length
    if(len>0) {
      array(0)(0) = value
    }
      for (j <- 1 to len - 1) {
          System.arraycopy(array(0), 0, array(0), j, if (len - j < j) (len - j) else j);
    }
    for ( i <-1 to array.length-1)
      System.arraycopy(array(0),0,array(i),0,len)
  }
// Conclucsion: Array fill is faster than fibonacci fill bcoz of run time optimizations.

  // Fibonacci copy + arraycopy is 2nd fastest
  def Fastones(rows :Int , cols:Int):Array[Array[Double]]= {
    val result = Array.ofDim[Double](rows,cols)
    byteFill(result,1.toDouble)
    result
  }
  // Fibonacci copy for every row is slowest
  def Slowones(rows :Int , cols:Int):Array[Array[Double]]= {
    val result = Array.ofDim[Double](rows,cols)
    result.map(obj=>byteFill(obj,1.toDouble))
  }
  // Array Fill for every row is 3rd fast
  def Fillones(rows :Int , cols:Int):Array[Array[Double]]= {
    val result = Array.ofDim[Double](rows,cols)
    //result.map(obj=>byteFill(obj,1.toDouble))
    // Fill each row with 1.0
    for ( row <-result)
      Arrays.fill(row, 1.toDouble);
    result
  }
  //Array fill and Arrys copy is the fastest
  def Fillsones(rows :Int , cols:Int):Array[Array[Double]]= {
    val result = Array.ofDim[Double](rows,cols)
    Arrays.fill(result(0), 1.toDouble);
    for ( i  <- 1 to rows - 1)
      System.arraycopy(result(0), 0, result(i), 0, cols)
    result
  }

  def main(args: Array[String]): Unit = {
    /*
    println("Hello, Scala developer!")
    val nums: List[Double] = List(3, 1, 3, 4)
    println(average(nums))
    var dm1: DenseMatrix = null
    val dm2: DenseMatrix = null
    dm1 = new DenseMatrix("1 2; 3 4") // create new matrix
    // with rows {1,2} and {3,4}
    dm1 = new DenseMatrix(Array[Array[Double]](Array(1, 2), Array(3, 4)))
    // with rows {1,2} and {3,4}
    dm1 = zeros(5, 3) // creates a dense matrix with 5 rows, and 3 columns
    dm1 = rand(5, 3) // create a 5*3 dense matrix filled with random numbers
    dm1 = ones(5, 3) // 5*3 matrix filled with '1's
    dm1 = diag(rand(5, 1)) // creates a 5*5 diagonal matrix of random numbers
    dm1 = eye(5) // creates a 5*5 identity matrix
    val o = new MnistNNet.NNet.test
    println(o.calc())
    var sm1: SparseMatrixLil = null
    sm1 = spzeros(5, 3) // creates an empty 5*3 sparse matrix
    sm1 = spdiag(rand(5, 1)) // creates a sparse 5*5 diagonal matrix of random
    // numbers
    sm1 = speye(5) // creates a 5*5 identity matrix, sparse
    //val t = new MnistNNet.NNet.test{}
    //println(t.calc())
    */


    //val url = new URL("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")
    //println(url.getPath().substring(url.getPath().lastIndexOf('/') + 1))
    //fileDownloader("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz","t10k-labels-idx1-ubyte.gz")
    //System.loadLibrary("mkl_rt")
    /*
    for ( i <-1 to 10 ) {
      val t = System.currentTimeMillis()
      val dm1: DenseMatrix = ones(2000, 1000)
      val dm2: DenseMatrix = DenseMatrix.rand(1000, 2000)
      val dm3: DenseMatrix = dm1.mmul(dm2)
      println((System.currentTimeMillis() - t) / 1000.0)
    }
*/
    /*
    System.loadLibrary("mkl_rt")
    for ( i <-1 to 10 ) {
      val t = System.currentTimeMillis()
      val dm1: INDArray = Nd4j.ones(2000, 1000);
      val dm2: INDArray = Nd4j.rand(1000, 2000);
      val dm3 = dm1.mmul(dm2)
      println((System.currentTimeMillis() - t) / 1000.0)

    }
    */
    /*
    System.loadLibrary("mkl_rt")
    val t =System.currentTimeMillis()
    val  dm1:INDArray = Nd4j.create(Fillsones(2000, 1000));
    val dm2:INDArray = Nd4j.create(rand(1000, 2000));
    val dm3  = dm1.mmul(dm2)
    println((System.currentTimeMillis() - t)/1000.0)
*/
    //println(Arrays.toString(result))
    /*
    for ( i <-1 to 1 ) {
      var t = System.currentTimeMillis()
      val dm: Array[Array[Double]] = Fastones(2000, 1000)
      println((System.currentTimeMillis() - t) / 1000.0)
      t = System.currentTimeMillis()
      val dm1: Array[Array[Double]] = Fillones(2000, 1000)
      println((System.currentTimeMillis() - t) / 1000.0)
      t = System.currentTimeMillis()
      val dm2: Array[Array[Double]] = Fillsones(2000, 1000)
      println((System.currentTimeMillis() - t) / 1000.0)
      t = System.currentTimeMillis()
      val dm3: Array[Array[Double]] = Slowones(2000, 1000)
      println((System.currentTimeMillis() - t) / 1000.0)
      println("----")
    }
*/

    /*

    {
      val dm1: Array[Array[Double]] = rand(10000, 4000)
      val dm2: Array[Array[Double]] = rand(10000, 1)
      var t = System.currentTimeMillis()
      MnistLoader.shuffle(dm1, dm2)
      println((System.currentTimeMillis() - t) / 1000.0)
    }
    {
      val dm1: Array[Array[Double]] = rand(10000, 4000)
      val dm2: Array[Array[Double]] = rand(10000, 1)
      val t = System.currentTimeMillis()
      MnistLoader.slowshuffle(dm1, dm2)
      println((System.currentTimeMillis() - t) / 1000.0)
   }
    */

    /*
    val b:IndexedSeq[Double] = 1.to(10).map(a => math.sqrt(a))
    val c:Vector[Int] = (1 to 10).toVector
    println(b)
    println(c)
    */

    MnistNNet.NNet.GenericMLP.run()

  }
}