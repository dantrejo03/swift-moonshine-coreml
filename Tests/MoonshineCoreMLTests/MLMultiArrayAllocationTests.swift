import CoreML
import Testing
@testable import MoonshineCoreML

struct MLMultiArrayAllocationTests {
    @Test func rowMajorStridesMultiplyTheLaterAxes() {
        #expect(MLMultiArray.rowMajorStrides(for: [6, 1, 8, 768, 64]) == [393_216, 393_216, 49_152, 64, 1])
        #expect(MLMultiArray.rowMajorStrides(for: [1280]) == [1])
    }

    @Test func zeroedArrayIsRowMajorAndAllZeros() throws {
        let array = try MLMultiArray.makeZeroed(shape: [2, 3, 5], dataType: .float32)
        #expect(array.integerShape == [2, 3, 5])
        #expect(array.integerStrides == [15, 5, 1])
        array.withUnsafeBufferPointer(ofType: Float.self) { numbers in
            #expect(numbers.prefix(30).allSatisfy { $0 == 0 })
        }
    }

    @Test func surfaceBackedArrayHasTheShapeAndIsAllZeros() throws {
        let shape = [2, 3, 5]
        let array = try MLMultiArray.makeZeroedSurfaceBackedFloat16(shape: shape)
        #expect(array.integerShape == shape)
        #expect(array.dataType == .float16)
        array.withUnsafeBufferPointer(ofType: Float16.self) { numbers in
            let strides = array.integerStrides
            for first in 0 ..< shape[0] {
                for second in 0 ..< shape[1] {
                    for third in 0 ..< shape[2] {
                        #expect(numbers[first * strides[0] + second * strides[1] + third * strides[2]] == 0)
                    }
                }
            }
        }
    }
}
