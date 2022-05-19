''' This file calculates the ratio r_LiPF6 for the standard electrolyte 1 M LiPF6 in EC:EMC (3:7 by weight) based on a literature value of density
and molar mass. '''

# Kremer, L.S., Danner, T., Hein, S., Hoffmann, A., Prifling, B., Schmidt, V., Latz, A., Wohlfahrt-Mehrens, M., 2020. Influence of the Electrolyte Salt Concentration on the Rate Capability of Ultra-Thick NCM 622 Electrodes. Batteries & Supercaps 3, 1172–1182. https://doi.org/10.1002/batt.202000098, Landesfeind, J., Gasteiger, H.A., 2019. Temperature and Concentration Dependence of the Ionic Transport Properties of Lithium-Ion Battery Electrolytes. J. Electrochem. Soc. 166, A3079. https://doi.org/10.1149/2.0571912jes
# https://www.sigmaaldrich.com/DE/de/product/aldrich/201146

def mass_ratio_standard_lipf6(rho_1M=1.2, M_LiPF6 = 151.91, solution_volume = 1.0, lipf6_concentration = 1.0, 
                              solvent_r_EC = 3, solvent_r_EMC = 7  ):
    """calculating the mass ratio of lipf6 for standard electroly 

    Args:
        rho_1M (float, optional): density of 1 M LiPF6 in EC:EMC 3:7 by weight in g/mL. Defaults to 1.2.
        M_LiPF6 (float, optional): molar mass of LiPF6 in g/mol. Defaults to 151.91.
        solution_volume (float, optional): calculations are performed for 1 L of the 1 M solution. Defaults to 1.
        lipf6_concentration (float, optional): concentration of the LiPF6 is 1 mol/L. Defaults to 1.
        solvent_r_EC (int, optional): ratio of EC is 3 by weight. Defaults to 3..
        solvent_r_EMC (int, optional): ratio of EMC is 7 by weight. Defaults to 7.
    """
    ## Get the mass of LiPF6
    # get the amount of substance (for sake of completeness)
    n_LiPF6 = lipf6_concentration * solution_volume
    # get the mass of LiPF6
    m_LiPF6 = n_LiPF6 * M_LiPF6
    ## Get the total mass of the solution
    m_total = (solution_volume * 1000.) * rho_1M  # volume needs to be transferred to mL
    ## Get the mass of the solvent in the solution
    m_solvent = m_total - m_LiPF6
    # get mass of EC
    m_EC = (solvent_r_EC / (solvent_r_EC + solvent_r_EMC)) * m_solvent
    # get mass of EMC
    m_EMC = (solvent_r_EMC / (solvent_r_EC + solvent_r_EMC)) * m_solvent
    ## Get the mass ratio of LiPF6
    r_LiPF6 = m_LiPF6 / m_EC
    return r_LiPF6