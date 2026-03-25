"""Generate ODE_system.h — mostly a static template."""


def generate_ode_system_h(namespace: str, class_name: str) -> str:
    return f"""#ifndef __CancerVCT_ODE__
#define __CancerVCT_ODE__

#include "../cvode/CVODEBase.h"
#include "QSPParam.h"
#include "QSP_enum.h"

namespace {namespace}{{

class {class_name} :
    public CVODEBase
{{
public:
    //! ODE right hand side
    static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);
    //! Root finding (for events)
    static int g(realtype t, N_Vector y, realtype *gout, void *user_data);
    static std::string getHeader();
    static void setup_class_parameters(QSPParam& param);

    // accessing class parameters
    static double get_class_param(unsigned int i);
    static void set_class_param(unsigned int i, double v);

    template<class Archive>
    static void classSerialize(Archive & ar, const unsigned int  version);

    static bool use_steady_state;
    static bool use_resection;
    static double _QSP_weight;
private:
    // parameters shared by all instances of this class
    static state_type _class_parameter;
public:
    {class_name}();
    {class_name}(const {class_name}& c);
    ~{class_name}();

    // read tolerance from parameter file
    void setup_instance_tolerance(QSPParam& param);
    // read variable values from parameter file
    void setup_instance_variables(QSPParam& param);
    // evaluate initial assignment
    void eval_init_assignment(void);

    unsigned int get_num_variables(void)const {{ return _species_var.size(); }};
    unsigned int get_num_params(void)const {{ return _class_parameter.size(); }};

protected:
    void setupVariables(void);
    void setupEvents(void);
    void initSolver(realtype t0);
    void update_y_other(void);
    bool triggerComponentEvaluate(int i, realtype t, bool curr);
    //! evaluate one event trigger
    bool eventEvaluate(int i);
    //! execute one event
    bool eventExecution(int i, bool delay, realtype& dt);
    //! unit conversion factor for species (y and non-y)
    realtype get_unit_conversion_species(int i) const;
    //! unit conversion factor for non-species variables
    realtype get_unit_conversion_nspvar(int i) const;

private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const unsigned int /*version*/);
}};

template<class Archive>
void {class_name}::classSerialize(Archive & ar, const unsigned int version){{
    ar & BOOST_SERIALIZATION_NVP(_class_parameter);
}}

template<class Archive>
void {class_name}::serialize(Archive & ar, const unsigned int  version){{
    ar & boost::serialization::base_object<CVODEBase>(*this);
}}

inline double {class_name}::get_class_param(unsigned int i){{
    if (i < _class_parameter.size())
    {{
        return _class_parameter[i];
    }}
    else{{
        throw std::invalid_argument("Accessing ODE class parameter: out of range");
    }}
}}

inline void {class_name}::set_class_param(unsigned int i, double v){{
    if (i < _class_parameter.size())
    {{
        _class_parameter[i] = v;
    }}
    else{{
        throw std::invalid_argument("Setting ODE class parameter: out of range");
    }}
}}

}};
#endif
"""
