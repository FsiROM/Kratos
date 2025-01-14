// KRATOS___
//     //   ) )
//    //         ___      ___
//   //  ____  //___) ) //   ) )
//  //    / / //       //   / /
// ((____/ / ((____   ((___/ /  MECHANICS
//
//  License:         geo_mechanics_application/license.txt
//
//  Main authors:    Anne van de Graaf,
//                   Marjan Fathian,
//                   Gennady Markelov
//

#include "custom_utilities/parameters_utilities.h" 

namespace Kratos {

Parameters ParametersUtilities::ExtractParameters(const Parameters& rSourceParameters,
                                                        const std::vector<std::string>& rNamesOfParametersToCopy)
{
    auto result = Parameters{};
    result.CopyValuesFromExistingParameters(rSourceParameters, rNamesOfParametersToCopy);
    return result;
 }

void ParametersUtilities::AppendParameterNameIfExists(const std::string& rParameterName,
    const Parameters& rSourceParameters,
    std::vector<std::string>& rResult)
{
    if (rSourceParameters.Has(rParameterName)) {
        rResult.emplace_back(rParameterName);
    }
}

bool ParametersUtilities::HasTableAttached(const Parameters& rSettings)
{
    if (rSettings["table"].IsNumber()) {
        return rSettings["table"].GetInt() != 0;
    }

    KRATOS_ERROR_IF_NOT(rSettings["table"].IsArray()) << "'table' is neither a single number nor an array of numbers";

    const auto& table = rSettings["table"];
    return std::any_of(table.begin(), table.end(),
        [](const auto& value) {return value.GetInt() != 0; });
}

bool ParametersUtilities::HasTableAttached(const Parameters& rSettings,
    int component)
{
    return rSettings["table"][component].GetInt() != 0;
}

}