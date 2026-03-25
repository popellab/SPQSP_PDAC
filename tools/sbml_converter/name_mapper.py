"""Map SBML element names to C++ enum names and identifiers."""

from sbml_parser import SBMLModel, Species, Parameter, Compartment


class NameMapper:
    """Maps SBML IDs/names to C++ enum names, variable names, and XML paths."""

    def __init__(self, model: SBMLModel):
        self.model = model
        # ID -> C++ name caches
        self._species_enum = {}   # SBML ID -> SP_xxx
        self._param_enum = {}     # SBML ID -> P_xxx
        self._qsp_enum = {}       # SBML ID -> QSP_xxx
        self._aux_var = {}        # SBML ID -> AUX_VAR_xxx
        self._xml_path = {}       # SBML ID -> XML path string
        self._display_name = {}   # SBML ID -> "Compartment.Species" display name

        self._build_mappings()

    def _sanitize(self, name: str) -> str:
        """Make a name safe for C++ identifiers."""
        return name.replace(".", "_").replace("-", "_").replace(" ", "_")

    def _species_display_name(self, sp: Species) -> str:
        """Get display name like 'V_T.CD8' for a species."""
        return f"{sp.compartment_name}.{sp.name}"

    def _build_mappings(self):
        m = self.model

        # Species enum names: SP_V_T_CD8
        for sp in m.sp_var + m.sp_other:
            display = self._species_display_name(sp)
            self._display_name[sp.id] = display
            enum_name = "SP_" + self._sanitize(display)
            self._species_enum[sp.id] = enum_name

        # Parameter class enum names: P_xxx
        # Compartments first (as parameters), then actual parameters
        param_idx = 0
        for comp in m.compartments:
            enum_name = "P_" + self._sanitize(comp.name)
            self._param_enum[comp.id] = enum_name
            self._display_name[comp.id] = comp.name
            param_idx += 1

        for param in m.p_const:
            enum_name = "P_" + self._sanitize(param.name)
            self._param_enum[param.id] = enum_name
            self._display_name[param.id] = param.name
            param_idx += 1

        # QSP param enum names (for XML parameter file loading)
        # Order: sim settings (hardcoded), compartments, species, parameters
        # Sim settings are hardcoded in the generator

        for comp in m.compartments:
            self._qsp_enum[comp.id] = "QSP_" + self._sanitize(comp.name)
            self._xml_path[comp.id] = f"Param.QSP.init_value.Compartment.{self._sanitize(comp.name)}"

        for sp in m.sp_var + m.sp_other:
            display_sanitized = self._sanitize(self._species_display_name(sp))
            self._qsp_enum[sp.id] = "QSP_" + display_sanitized
            self._xml_path[sp.id] = f"Param.QSP.init_value.Species.{display_sanitized}"

        for param in m.p_const:
            self._qsp_enum[param.id] = "QSP_" + self._sanitize(param.name)
            self._xml_path[param.id] = f"Param.QSP.init_value.Parameter.{self._sanitize(param.name)}"

        # Assignment rule variables: AUX_VAR_xxx
        for ar in m.assignment_rules:
            self._aux_var[ar.variable_id] = "AUX_VAR_" + self._sanitize(ar.variable_name)

    # ---- Public API ----

    def species_enum(self, sbml_id: str) -> str:
        """Get SP_xxx enum name for a species."""
        return self._species_enum[sbml_id]

    def param_enum(self, sbml_id: str) -> str:
        """Get P_xxx enum name for a class parameter."""
        return self._param_enum[sbml_id]

    def qsp_enum(self, sbml_id: str) -> str:
        """Get QSP_xxx enum name for an XML parameter."""
        return self._qsp_enum[sbml_id]

    def aux_var(self, sbml_id: str) -> str:
        """Get AUX_VAR_xxx name for an assignment rule variable."""
        return self._aux_var[sbml_id]

    def xml_path(self, sbml_id: str) -> str:
        """Get XML path for a parameter (e.g., 'Param.QSP.init_value.Species.V_T_CD8')."""
        return self._xml_path[sbml_id]

    def display_name(self, sbml_id: str) -> str:
        """Get human-readable display name (e.g., 'V_T.CD8')."""
        return self._display_name.get(sbml_id, sbml_id)

    def is_species(self, sbml_id: str) -> bool:
        return sbml_id in self.model.id_to_species

    def is_parameter(self, sbml_id: str) -> bool:
        return sbml_id in self.model.id_to_parameter

    def is_compartment(self, sbml_id: str) -> bool:
        return sbml_id in self.model.id_to_compartment

    def is_assignment_rule_target(self, sbml_id: str) -> bool:
        return sbml_id in self._aux_var

    def is_ode_species(self, sbml_id: str) -> bool:
        sp = self.model.id_to_species.get(sbml_id)
        return sp is not None and sp.is_ode_var

    def is_other_species(self, sbml_id: str) -> bool:
        sp = self.model.id_to_species.get(sbml_id)
        return sp is not None and sp.is_other

    def species_index(self, sbml_id: str) -> int:
        """Get index of species in sp_var or sp_other list."""
        for i, sp in enumerate(self.model.sp_var):
            if sp.id == sbml_id:
                return i
        for i, sp in enumerate(self.model.sp_other):
            if sp.id == sbml_id:
                return len(self.model.sp_var) + i
        raise KeyError(f"Species {sbml_id} not found in sp_var or sp_other")

    def param_index(self, sbml_id: str) -> int:
        """Get index in the class parameter vector."""
        # Compartments first
        for i, comp in enumerate(self.model.compartments):
            if comp.id == sbml_id:
                return i
        offset = len(self.model.compartments)
        for i, param in enumerate(self.model.p_const):
            if param.id == sbml_id:
                return offset + i
        raise KeyError(f"Parameter {sbml_id} not found in p_const")

    def resolve_name(self, sbml_id: str, context: str = "f") -> str:
        """Resolve an SBML ID to its C++ expression in the given context.

        Contexts:
          'f'     - inside f() function: SPVAR(SP_xxx), PARAM(P_xxx), AUX_VAR_xxx
          'event' - inside event functions: NV_DATA_S(_y)[SP_xxx], _class_parameter[P_xxx]
          'other' - inside update_y_other(): same as event
        """
        # Assignment rule variables
        if sbml_id in self._aux_var:
            return self._aux_var[sbml_id]

        # Species
        if sbml_id in self.model.id_to_species:
            sp = self.model.id_to_species[sbml_id]
            if sp.is_ode_var:
                enum = self._species_enum[sbml_id]
                if context == "f":
                    return f"SPVAR({enum})"
                else:
                    return f"NV_DATA_S(_y)[{enum}]"
            elif sp.is_other:
                idx = 0
                for i, s in enumerate(self.model.sp_other):
                    if s.id == sbml_id:
                        idx = i
                        break
                return f"_species_other[{idx}]"

        # Compartments (as class parameters)
        if sbml_id in self._param_enum:
            enum = self._param_enum[sbml_id]
            if context == "f":
                return f"PARAM({enum})"
            else:
                return f"_class_parameter[{enum}]"

        # Parameters (as class parameters)
        if sbml_id in self._param_enum:
            enum = self._param_enum[sbml_id]
            if context == "f":
                return f"PARAM({enum})"
            else:
                return f"_class_parameter[{enum}]"

        # Fallback: just use the name
        return sbml_id
