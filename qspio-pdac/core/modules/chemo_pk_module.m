% Cytotoxic Chemotherapy PK-PD Module (parametrized, per-drug)
%
% Faithful per-drug population PK with an Emax cell-kill PD, for the PASS-01
% regimens (mFOLFIRINOX: 5-FU, oxaliplatin, irinotecan/SN-38; GnP: gemcitabine).
% Nab-paclitaxel keeps its own dedicated module (nabpaclitaxel_module.m).
%
%   model = chemo_pk_module(model, name, params, n_periph, elim, metabolite)
%
%   name       drug tag: 'gem' | 'fu5' | 'oxa' | 'iri'
%   n_periph   number of peripheral PK compartments (0,1,2)
%   elim       central elimination of the PARENT:
%                'linear' -> CL_<name>            (first-order)
%                'mm'     -> Vmax_<name>,Km_<name> (Michaelis-Menten; 5-FU/DPD)
%   metabolite '' (none) or a tag (e.g. 'sn38'): adds a 1-compartment active
%              metabolite formed from the parent central compartment at molar
%              fraction Fm_<name>; the metabolite becomes the cytotoxic species.
%
% Units convention (mass-based -> literature popPK values plug in verbatim,
% and milligram doses need no MW conversion):
%   species        milligram/liter    (plasma / tissue concentration)
%   compartments   liter              (apparent volumes of distribution)
%   CL, Q          liter/hour
%   Vmax           milligram/hour ; Km  milligram/liter
%   IC50           milligram/liter    (PD; calibrated, nominal default)
%   k_kill         1/day              (PD max kill rate; calibrated)
%
% Dosing (zero-order IV infusion on the appropriate schedule) is attached
% externally via schedule_dosing.m onto the central species '<name>' in its
% central compartment 'C_<name>' (TargetName), so the same module serves every
% regimen/schedule. See drug_metadata.yaml for the C++-side dose mapping.
%
% PD: Emax kill of V_T.C1 by the cytotoxic species' TUMOR concentration
%     (plasma->tumor penetration ratio r_<name>), mirroring nabpaclitaxel_module:
%       rate = k_kill_<name> * V_T.C1 * D_T/(D_T+IC50_<name>) * min(C_total,Kc)/C_total
%     The kill routes cells into V_T.C_x (dead pool) -> drives the CA19-9 shed spike.
%
% Dependencies: cancer_module (V_T.C1, C_total, V_T.C_x), simbio_init (V_C..V_LN).

function model = chemo_pk_module(model, name, params, n_periph, elim, metabolite)

	if nargin < 4 || isempty(n_periph),  n_periph = 1;        end
	if nargin < 5 || isempty(elim),      elim = 'linear';     end
	if nargin < 6,                       metabolite = '';     end

	pv = @(f) params.(f).Value;            % value helper
	pu = @(f) params.(f).Units;            % units helper
	function p = addp(pname)               % add a model parameter from params.<pname>
		p = addparameter(model, pname, pv(pname), 'ValueUnits', pu(pname), ...
			'ConstantValue', false, 'Notes', params.(pname).Notes);
		p.UserData.Source = params.(pname).Notes;
		p.UserData.derivedFrom = params.(pname).derivedFrom;
	end

	% ---- Central compartment + parent species ----
	V1 = ['V_' name '_1'];  addp(V1);
	compC = addcompartment(model, ['C_' name], pv(V1), 'CapacityUnits', pu(V1), ...
		'ConstantCapacity', false, 'Notes', ['Central PK compartment for ' name]);
	addspecies(compC, name, 0, 'InitialAmountUnits', 'milligram/liter', ...
		'Notes', [name ' concentration in central compartment']);

	% ---- Peripheral compartments + linear exchange ----
	for j = 1:n_periph
		Vp = ['V_' name '_' num2str(j+1)];   addp(Vp);
		Qp = ['Q_' name '_' num2str(j+1)];   addp(Qp);
		compP = addcompartment(model, ['P' num2str(j) '_' name], pv(Vp), ...
			'CapacityUnits', pu(Vp), 'ConstantCapacity', false, ...
			'Notes', ['Peripheral PK compartment ' num2str(j) ' for ' name]);
		addspecies(compP, name, 0, 'InitialAmountUnits', 'milligram/liter', ...
			'Notes', [name ' concentration in peripheral compartment ' num2str(j)]);
		% Diffusive exchange as TWO one-way reactions with microconstants
		% (k12 = Q/Vc, k21 = Q/Vp). This is mass-correct for unequal
		% compartment volumes AND avoids a single sign-changing reversible
		% rate, which stalls the stiff CVODE solver after a dose bolus.
		r = addreaction(model, ['C_' name '.' name ' -> P' num2str(j) '_' name '.' name]);
		set(r, 'ReactionRate', [Qp '/' V1 '*C_' name '.' name], ...
			'Notes', ['Distribution of ' name ' central->P' num2str(j)]);
		r = addreaction(model, ['P' num2str(j) '_' name '.' name ' -> C_' name '.' name]);
		set(r, 'ReactionRate', [Qp '/' Vp '*P' num2str(j) '_' name '.' name], ...
			'Notes', ['Redistribution of ' name ' P' num2str(j) '->central']);
	end

	% ---- Central elimination (parent) ----
	switch lower(elim)
		case 'mm'
			addp(['Vmax_' name]);  addp(['Km_' name]);
			r = addreaction(model, ['C_' name '.' name ' -> null']);
			set(r, 'ReactionRate', ...
				['Vmax_' name '/' V1 '*(C_' name '.' name '/(Km_' name ' + C_' name '.' name '))'], ...
				'Notes', ['Nonlinear (Michaelis-Menten) elimination of ' name ' from central']);
			CLname = '';
		otherwise % linear
			CLname = ['CL_' name];  addp(CLname);
			r = addreaction(model, ['C_' name '.' name ' -> null']);
			set(r, 'ReactionRate', [CLname '/' V1 '*C_' name '.' name], ...
				'Notes', ['Linear clearance of ' name ' from central']);
	end

	% ---- Active metabolite (e.g. irinotecan -> SN-38) ----
	cyto = name;                               % default cytotoxic species = parent
	cytoV1 = V1;                               % its central volume param
	cytoComp = ['C_' name];
	if ~isempty(metabolite)
		m = metabolite;
		Vm  = ['V_' m '_1'];      addp(Vm);
		CLm = ['CL_' m];          addp(CLm);
		Fm  = ['Fm_' name];       addp(Fm);
		MWr = ['MWr_' m];         addp(MWr);   % MW(metabolite)/MW(parent), for mass->mass conversion
		compM = addcompartment(model, ['C_' m], pv(Vm), 'CapacityUnits', pu(Vm), ...
			'ConstantCapacity', false, 'Notes', ['Central PK compartment for metabolite ' m]);
		addspecies(compM, m, 0, 'InitialAmountUnits', 'milligram/liter', ...
			'Notes', [m ' (active metabolite of ' name ') concentration']);
		% Formation from parent central (mass: Fm*CL_parent*C_parent*MWratio, into metabolite volume)
		if isempty(CLname), CLterm = ['Vmax_' name '/' V1 '*(C_' name '.' name '/(Km_' name ' + C_' name '.' name '))*' V1];
		else,               CLterm = [CLname '*C_' name '.' name]; end
		r = addreaction(model, ['null -> C_' m '.' m]);
		set(r, 'ReactionRate', [Fm '*' MWr '*(' CLterm ')/' Vm], ...
			'Notes', ['Formation of ' m ' from ' name ' (fraction Fm of parent clearance)']);
		% Metabolite elimination
		r = addreaction(model, ['C_' m '.' m ' -> null']);
		set(r, 'ReactionRate', [CLm '/' Vm '*C_' m '.' m], ...
			'Notes', ['Linear clearance of metabolite ' m]);
		cyto = m;  cytoV1 = Vm;  cytoComp = ['C_' m];
	end

	% ---- PD: tumor penetration + Emax kill of V_T.C1 ----
	addp(['r_' name]);        % plasma->tumor penetration ratio (dimensionless)
	addp(['IC50_' name]);     % half-maximal tumor conc for kill (mg/L)
	addp(['k_kill_' name]);   % max kill rate (1/day)
	addp(['Kc_' name]);       % half-max cancer number for cytotoxic diffusion (cell)
	% Tumor concentration of the cytotoxic species, as a PARAMETER via assignment
	% rule (NOT a species): a rule-assigned concentration species in the
	% variable-volume V_T compartment trips the codegen's amount<->conc volume
	% conversion (undeclared AUX_VAR_V_T). A parameter is algebraically identical
	% (Ct = r * plasma) and avoids that entirely.
	CtT = ['Ct_' name];
	addparameter(model, CtT, 0, 'ValueUnits', 'milligram/liter', 'ConstantValue', false, ...
		'Notes', [cyto ' concentration in tumor (r_' name ' x plasma)']);
	addrule(model, [CtT ' = r_' name ' * ' cytoComp '.' cyto], 'repeatedAssignment');
	% Emax kill -> dead pool (drives CA19-9 shed)
	r = addreaction(model, 'V_T.C1 -> V_T.C_x');
	set(r, 'ReactionRate', ...
		['k_kill_' name '*V_T.C1*(' CtT '/(' CtT ' + IC50_' name '))*min(C_total,Kc_' name ')/C_total'], ...
		'Notes', ['Cancer cell kill by ' name ' (' cyto ', Emax)']);

	% ---- Resistant clone C2: same Emax kill, IC50 raised r_resist-fold (cross-drug) ----
	% r_resist is shared across all cytotoxic modules; add it once if absent.
	if isempty(sbioselect(model,'Type','parameter','Name','r_resist'))
		r_res = addparameter(model,'r_resist',params.r_resist.Value,'ValueUnits',params.r_resist.Units,'ConstantValue',false, ...
			'Notes','Fold-increase of cytotoxic IC50 in the resistant clone C2 (cross-drug)');
		r_res.UserData.Source = params.r_resist.Notes; r_res.UserData.derivedFrom = params.r_resist.derivedFrom;
	end
	r = addreaction(model, 'V_T.C2 -> V_T.C_x');
	set(r, 'ReactionRate', ...
		['k_kill_' name '*V_T.C2*(' CtT '/(' CtT ' + IC50_' name '*r_resist))*min(C_total,Kc_' name ')/C_total'], ...
		'Notes', ['Resistant-clone (C2) kill by ' name ' (' cyto ', Emax, IC50 x r_resist)']);

end
