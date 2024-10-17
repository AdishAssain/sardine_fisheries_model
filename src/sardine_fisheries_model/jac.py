import numpy as np


def jacobian_A(params):
    jacobian_A = np.array([
        [params['a_SN'] * (params['b_SRIA'] * params['EE_SRIA_SN'] - (1 - params['b_SMN']) * params['EE_SC_SN'] - params['b_SMN'] * params['EE_SMN_SN']) / params['b_SMN'],
         (params['a_SN'] / params['b_SMN']) * (params['b_SRIA'] * params['EE_SRIA_JFN'] + (1 - params['b_SRIA']) * params['EE_MN_JFN'] - (1 - params['b_SMN']) * params['EE_SC_JFN'] - params['b_SMN'] * params['EE_SMN_JFN']),
         params['a_SN'] * ((1 - params['b_SMN']) * params['EE_SC_FG']) / params['b_SMN'],
         params['a_SN'] * ((1 - params['b_SMN']) * params['EE_SC_FT']) / params['b_SMN']],

        [params['a_JFN'] * params['b_JFG'] * params['EE_JFG_SN'] / (1 - params['b_JFPM']),
         (params['a_JFN'] / (1 - params['b_JFPM'])) * ((1 - params['b_JFG']) * params['EE_JFMS_JFN'] - params['b_JFPM'] * params['EE_JFPM_JFN'] - (1 - params['b_JFPM']) * params['EE_JFS_JFN']),
         0, 0],

        [params['a_FG'] * (params['b_FGOBRS'] * params['EE_FGOBRS_SN'] + (1 - params['b_FGOBRS']) * params['EE_FGIBRS_SN'] - params['EE_GE_SN']),
         0,
         params['a_FG'] * (params['b_FGOBRS'] * params['EE_FGOBRS_FG'] + (1 - params['b_FGOBRS']) * params['EE_FGIBRS_FG'] - params['EE_GE_FG']),
         params['a_FG'] * (params['b_FGOBRS'] * params['EE_FGOBRS_FT'] + (1 - params['b_FGOBRS']) * params['EE_FGIBRS_FT'] - params['EE_GE_FT'])],

        [params['a_FT'] * (params['EE_FT_SN'] - params['b_FCT'] * params['EE_FCT_SN']),
         0,
         params['a_FT'] * (params['EE_FT_FG'] - params['b_FCT'] * params['EE_FCT_FG']),
         params['a_FT'] * (1 - params['b_FCT'] * params['EE_FCT_FT'] - (1 - params['b_FCT']) * params['EE_LFT_FT'])]
    ])

    return jacobian_A


def jacobian_B(params):
    """
    Construct the Jacobian matrix using the provided parameters.
    This matrix will be used for stability or sensitivity analysis.
    """
    a_SN = params['a_SN']
    a_JFN = params['a_JFN']
    a_FG = params['a_FG']
    a_FT = params['a_FT']
    b_SRIA = params['b_SRIA']
    b_SMN = params['b_SMN']
    b_JFG = params['b_JFG']
    b_JFPM = params['b_JFPM']
    b_FGOBRS = params['b_FGOBRS']
    b_FCT = params['b_FCT']

    jacobian_matrix = np.array([
        [-a_SN, b_SRIA, 0, 0],
        [a_JFN, -b_SMN, a_FG, 0],
        [0, b_JFG, -b_FGOBRS, a_FT],
        [0, 0, b_FCT, -b_JFPM]
    ])

    return jacobian_matrix


def calculate_eigenvalues(jac_matrix):
    """
    Calculate the dominant (largest) eigenvalue from the Jacobian matrix.
    """
    eigenvalues = np.linalg.eigvals(jac_matrix)
    dominant_eigenvalue = np.real(np.max(np.real(eigenvalues)))
    return dominant_eigenvalue, eigenvalues


def jacobian_diagnostics(jac_matrix):
    """
    Function to calculate the determinant, trace, and condition number of the Jacobian matrix
    """
    determinant = np.linalg.det(jac_matrix)
    trace = np.trace(jac_matrix)
    condition_number = np.linalg.cond(jac_matrix)
    return determinant, trace, condition_number
