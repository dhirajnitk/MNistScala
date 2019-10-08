package MnistNNet.NNet
import jeigen._
import java.lang._
trait CostFunction{
  def totalCost(activation : DenseMatrix, output: DenseMatrix) : Float
  def delta(z : DenseMatrix, activation: DenseMatrix, output: DenseMatrix) : DenseMatrix

}

trait QuadraticCost extends CostFunction with TransferFunction{
  override def totalCost(activation : DenseMatrix, output: DenseMatrix) : Float ={
    val costMat: DenseMatrix = activation.sub(output).pow(2).sum().pow(0.5)
    (costMat.getValues.foldLeft(0.0D)(_+_)/(2*costMat.rows)).toFloat
  }

}

trait ReLUQuadraticCost extends CostFunction with ReLUTransferFunction with QuadraticCost {
  override def delta(z : DenseMatrix, activation: DenseMatrix, output: DenseMatrix) : DenseMatrix = {
    activation.sub(output).mul(transferredSlope(z))
  }
}
trait SigmoidQuadraticCost extends CostFunction with SigmoidTransferFunction with QuadraticCost {
  override def delta(z : DenseMatrix, activation: DenseMatrix, output: DenseMatrix) : DenseMatrix = {
    activation.sub(output).mul(transferredSlope(z))
  }
}
trait TanhQuadraticCost extends CostFunction with TanhTransferFunction with QuadraticCost {
  override def delta(z : DenseMatrix, activation: DenseMatrix, output: DenseMatrix) : DenseMatrix = {
    // Speeds up learning in outer layer by projecting Tanh into (0,1) suitable only for xavier weights
    //activation.add(1).div(2).sub(output).mul(transferredSlope(z))
    //Normal Tanh configuration
     activation.sub(output).mul(transferredSlope(z))
  }
}

trait SoftMaxQuadraticCost extends CostFunction with SoftMaxTransferFunction with QuadraticCost {

  override def delta(z : DenseMatrix, activation: DenseMatrix, output: DenseMatrix) : DenseMatrix = {
    val left = activation.sub(output)
    val right = transferredSlope(z)
    flatMult(left, right)
  }
}

trait CrossEntropyCost extends CostFunction  with TransferFunction {
  override def totalCost(activation: DenseMatrix, output: DenseMatrix): Float =
  {
    //ArrayXf m1 = (-y.array().cwiseProduct(a.array().log()) - (1 - y.array()).cwiseProduct((1 - a.array()).log()));
    //return (m1 != m1).select(0, m1).sum()/m1.rows();
    var costMat: DenseMatrix = output.neg().mul(activation.log()).sub(output.neg().add(1).mul(activation.neg().add(1).log()))
    (costMat.getValues.foldLeft(0.0D)((r, c) => if (c.isNaN()) r + c else r) / (costMat.rows)).toFloat

  }

}
trait SigmoidCrossEntropyCost extends CostFunction  with TransferFunction with CrossEntropyCost{
  this: SigmoidTransferFunction  =>
    override def delta(z: DenseMatrix, activation: DenseMatrix, output: DenseMatrix): DenseMatrix =
    {
      activation.sub(output)
    }


}

trait TanhCrossEntropyCost extends CostFunction with TransferFunction {

  this: TanhTransferFunction  =>
  override def totalCost(activation: DenseMatrix, output: DenseMatrix): Float =
  {
    var costMat:DenseMatrix = output.add(1).div(2).neg().mul(activation.add(1).div(2).log()).add(
      output.neg().add(1).div(2).mul(activation.neg().add(1).div(2).log()))
    (costMat.getValues.foldLeft(0.0D)((r, c) => if (c.isNaN()) r + c else r) / (costMat.rows)).toFloat
  }
  override def delta(z : DenseMatrix, activation: DenseMatrix, output: DenseMatrix) : DenseMatrix = {
     // The True delta requires output in [-1, 1]
    //activation.mul(output).sub(1)
    // Instead we can transform/scale activation to emulate same behavior for tanh in [0,1]
    activation.add(1).div(2).sub(output)

  }
}

trait SoftMaxTransferLogLHoodCost extends CostFunction with TransferFunction {
  this : SoftMaxTransferFunction =>
  override def totalCost(activation : DenseMatrix, output: DenseMatrix) : Float = {
    // ArrayXf m1 = -a.array().log().cwiseProduct(y.array());
    //return (m1 != m1).select(0, m1).sum()/m1.rows();
    val costMat: DenseMatrix = activation.log().mul(output).neg()
    (costMat.getValues.foldLeft(0.0D)((r,c) =>  if(c.isNaN()) r + c else r )/(costMat.rows)).toFloat
  }

  override def delta(z : DenseMatrix, activation: DenseMatrix, output: DenseMatrix) : DenseMatrix = {
    activation.sub(output)
  }
}

trait SigmoidTransferLogLHoodCost extends CostFunction with TransferFunction {
  this: SigmoidTransferFunction =>
  override def totalCost(activation : DenseMatrix, output: DenseMatrix) : Float = {
    // ArrayXf m1 = -a.array().log().cwiseProduct(y.array());
    //return (m1 != m1).select(0, m1).sum()/m1.rows();
    val costMat: DenseMatrix = activation.log().mul(output).neg()
    (costMat.getValues.foldLeft(0.0D)((r,c) =>  if(c.isNaN()) r + c else r )/(costMat.rows)).toFloat
  }

  override def delta(z : DenseMatrix, activation: DenseMatrix, output: DenseMatrix) : DenseMatrix = {
    activation.sub(output)
  }
}
