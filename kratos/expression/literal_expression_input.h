//    |  /           |
//    ' /   __| _` | __|  _ \   __|
//    . \  |   (   | |   (   |\__ `
//   _|\_\_|  \__,_|\__|\___/ ____/
//                   Multi-Physics
//
//  License:         BSD License
//                   Kratos default license: kratos/license.txt
//
//  Main authors:    Suneth Warnakulasuriya
//

#pragma once

// System incldues
#include <variant>

// Project includes
#include "containers/array_1d.h"
#include "expression/container_expression.h"
#include "expression/traits.h"
#include "containers/variable.h"
#include "expression_io.h"
#include "includes/define.h"
#include "includes/model_part.h"

namespace Kratos {


class KRATOS_API(KRATOS_CORE) LiteralExpressionIO
{
public:
    ///@name Type definitions
    ///@{

    using DataType = std::variant<
                                int,
                                double,
                                array_1d<double, 3>,
                                array_1d<double, 4>,
                                array_1d<double, 6>,
                                array_1d<double, 9>,
                                Vector,
                                Matrix>;

    using VariableType = std::variant<
                                const Variable<int>*,
                                const Variable<double>*,
                                const Variable<array_1d<double, 3>>*,
                                const Variable<array_1d<double, 4>>*,
                                const Variable<array_1d<double, 6>>*,
                                const Variable<array_1d<double, 9>>*,
                                const Variable<Vector>*,
                                const Variable<Matrix>*>;

    ///}
    ///@name Public classes
    ///@{

    class KRATOS_API(KRATOS_CORE) LiteralExpressionInput : public ExpressionInput
    {
    public:
        ///@name Type definitions
        ///@{

        KRATOS_CLASS_POINTER_DEFINITION(LiteralExpressionInput);

        ///@}
        ///@name Life cycle
        ///@{

        LiteralExpressionInput(
            const ModelPart& rModelPart,
            const DataType& rValue,
            const ContainerType& rContainerType,
            const MeshType& rMeshType = MeshType::Local);

        ~LiteralExpressionInput() override = default;

        ///@}
        ///@name Public operations
        ///@{

        Expression::Pointer Execute() const override;

        ///@}

    private:
        ///@name Private member variables
        ///@{

        const ModelPart& mrModelPart;

        const DataType mValue;

        const ContainerType mContainerType;

        const MeshType mMeshType;

        ///@}

    };

    ///@}
    ///@name Public static operations
    ///@{

    template<class TContainerType, MeshType TMeshType = MeshType::Local>
    static void SetData(
        ContainerExpression<TContainerType, TMeshType>& rContainerExpression,
        const DataType& rValue);

    template<class TContainerType, MeshType TMeshType = MeshType::Local>
    static void SetDataToZero(
        ContainerExpression<TContainerType, TMeshType>& rContainerExpression,
        const VariableType& rVariable);

    ///@}

}; // class ExpressionIO

} // namespace Kratos
