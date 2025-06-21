import numpy as np
from load_data import impact_factors, unified_emission_factors
from load_data import epyc_7b12_specs, epyc_7443_specs, epyc_9b14_specs
from modeling import calculate_total_impact, midpoint_to_endpoint, distance

epyc_7443_results = {
    'num_cores': 24,
    'compress-single':{
        'power': 76.93,
        'throughput': [
            [13.64,13.66],
            [4481.6,4482.8],
            [16.1,16.2],
            [1441.9,1443.3],
            [36.064],
            # [207.09],
            [20.002],
            [9.215],
            [2.851],
            [1563.890896]
        ],
        'result_type': ['higher', 'higher', 'higher', 'higher', 'lower', 'lower', 'lower', 'lower', 'lower'],
    },
    'compress-multi':{
        'power': 111.67,
        'throughput': [
            [4.0123]
        ],
        'result_type': ['lower'],
    },
    'compile': {
        'power': 192.53,
        'throughput': [
            [60.967],
            [718.909],
            [np.nan]
        ],
        'result_type': [ 'lower', 'lower', 'lower'],
    },
    'fftw': {
        'power': 75.89,
        'throughput': [
            [27405],
        ],
        'result_type': ['higher'],
    },
    'jpeg': {
        'power': 115.62,
        'throughput': [
            [np.nan],
            [22.1857],
            [np.nan],
            [np.nan]
        ],
        'result_type': ['higher', 'higher', 'higher', 'higher'],
    },
    'npb':{
        'power': 167.66,
        'throughput': [
            [64224.98, 64374.75],
            [17040.24, 17250.77],
            [2481.79],
            [2419.86],
            [41783.27],
            [2034.11],
            [71422.35],
            [44071.27],
            [37581.68],
            [7807.08]
        ],
        'result_type': ['higher']*10
    },
    'ssl': {
        'power': 200.56,
        'throughput': [
            [34673721240],
            [11470254270],
            [6510.0],
            [424575.7],
            [np.nan],
            [np.nan],
            [np.nan],
            [np.nan],
        ],
        'result_type': ['higher']*8
    },
    'video': {
        'power': 135.88,
        'throughput': [
            [63.5611],
            [203.9333],
            [29.9943],
            [75.01]
        ],
        'result_type': ['higher']*4
    },
    'spark': {
        'power': 185.7,
        'throughput': [
            [18.75],
            [59.91],
            [(4.02*2+4.03+3.95)/4],
            [17.25],
            [12.8375],
            [15.0355],
            [13.8250]
        ],
        'result_type': ['lower']*7,
    }
}

epyc_7b12_results = {
    'num_cores': 32,
    'compress-single':{
        'power': 46.7,
        'throughput': [
            [11.03],
            [3606.2,3614],
            [11.9,12.7],
            [1139.9,1154.0],
            [45.515],
            # [207.09],
            [25.191],
            [11.483],
            [3.586],
            [1964.0236]
        ],
        'result_type': ['higher', 'higher', 'higher', 'higher', 'lower', 'lower', 'lower', 'lower', 'lower'],
    },
    'compress-multi':{
        'power': 65.83,
        'throughput': [
            [3.6552]
        ],
        'result_type': ['lower'],
    },
    'compile': {
        'power': 123.71,
        'throughput': [
            [55.579],
            [572.592],
            [np.nan]
        ],
        'result_type': [ 'lower', 'lower', 'lower'],
    },
    'fftw': {
        'power': 42.84,
        'throughput': [
            [21493,21676],
        ],
        'result_type': ['higher'],
    },
    'jpeg': {
        'power': 64.34,
        'throughput': [
            [np.nan],
            [24.42],
            [np.nan],
            [np.nan]
        ],
        'result_type': ['higher', 'higher', 'higher', 'higher'],
    },
    'npb':{
        'power': 76.48,
        'throughput': [
            [70975.47],
            [14421.59],
            [2639.08],
            [2678.76],
            [58199.11],
            [1409.96],
            [np.nan],
            [np.nan],
            [np.nan],
            [np.nan]
        ],
        'result_type': ['higher']*10
    },
    'ssl': {
        'power': 136.84,
        'throughput': [
            [40636784590],
            [13658399440],
            [7511.9],
            [487739.5],
            [np.nan],
            [np.nan],
            [np.nan],
            [np.nan],
        ],
        'result_type': ['higher']*8
    },
    'video': {
        'power': 58.37,
        'throughput': [
            [51.9183],
            [159.81],
            [np.nan],
            [np.nan]
        ],
        'result_type': ['higher']*4
    },
    'spark': {
        'power': 98.56,
        'throughput': [
            [26.3425],
            [48.4025],
            [3.835],
            [24.2725],
            [19.935],
            [20.7125],
            [20.29]
        ],
        'result_type': ['lower']*7,
    }
}

epyc_9b14_results = {
    'num_cores': 30,
    'compress-single':{
        'power': 39.4,
        'throughput': [
            [11.595],
            [3993.2,3993.8],
            [14.6],
            [1297.5,1303.6],
            [43.232],
            # [255.988],
            [22.400],
            [9.728],
            [3.313],
            [1811.486739]
        ],
        'result_type': ['higher', 'higher', 'higher', 'higher', 'lower', 'lower', 'lower', 'lower', 'lower'],
    },
    'compress-multi':{
        'power': 55.67,
        'throughput': [
            [3.6552]
        ],
        'result_type': ['lower'],
    },
    'compile': {
        'power': 100.48,
        'throughput': [
            [44.695],
            [464.457],
            [304.078]
        ],
        'result_type': [ 'lower', 'lower', 'lower'],
    },
    'fftw': {
        'power': 38.3,
        'throughput': [
            [20969],
        ],
        'result_type': ['higher'],
    },
    'jpeg': {
        'power': 54.68,
        'throughput': [
            [np.nan],
            [27.956],
            [52.846],
            [np.nan]
        ],
        'result_type': ['higher', 'higher', 'higher', 'higher'],
    },
    'npb':{
        'power': 58.53,
        'throughput': [
            [93997.06,94063.2],
            [18620.88,19007.21],
            [3687.57,3767.07],
            [3769.37,3786.53],
            [37666.28,37895.81],
            [2505.37],
            [72393.78],
            [42516.18],
            [56173.32],
            [39788.86]
        ],
        'result_type': ['higher']*10
    },
    'ssl': {
        'power': 122.42,
        'throughput': [
            [43958374593],
            [14611436017],
            [19975.6],
            [486743.7],
            [np.nan],
            [np.nan],
            [np.nan],
            [np.nan],
        ],
        'result_type': ['higher']*8
    },
    'video': {
        'power': 55.1,
        'throughput': [
            [76.0643],
            [214.0550],
            [31.1850],
            [84.772]
        ],
        'result_type': ['higher']*4
    },
    'spark': {
        'power': 92.09,
        'throughput': [
            [18],
            [44.03],
            [(3.21*2+3.22+3.17)/4],
            [16.2675],
            [12.8025],
            [14.38],
            [13.4275]
        ],
        'result_type': ['lower']*7,
    }
}

def revise_power_gcp(power_list, tdp, vm_cores, host_cores, alpha=1.15):
    """
    Revise power values based on TDP and core counts
    """
    vm_to_host_ratio = vm_cores / host_cores
    revised_power = []
    for power in power_list:
        tmp = power / vm_to_host_ratio
        tmp -= tdp*0.3
        tmp /= tdp*0.7
        # print(tmp)
        u_pkg = np.power(tmp, 1/alpha)
        u_vm = u_pkg / vm_to_host_ratio
        # print(f"u_pkg: {u_pkg:.4f}, u_vm: {u_vm:.4f}")
        revised_power.append(vm_to_host_ratio * ( tdp*0.3 + tdp*0.7 * u_vm**alpha))

    return revised_power

tests = sorted(epyc_7b12_results.keys())
original_power = {
    'EPYC 7B12': [epyc_7b12_results[test]['power'] for test in tests if test !='num_cores'],
    'EPYC 7443': [epyc_7443_results[test]['power'] for test in tests if test !='num_cores'],
    'EPYC 9B14': [epyc_9b14_results[test]['power'] for test in tests if test !='num_cores'],
}

def consolidate_list_sequences(seq):
    """
    Consolidate list sequences into a single list
    """
    consolidated = []
    for item in seq:
        if isinstance(item, list):
            consolidated.append(np.mean(item, axis=0))
        else:
            consolidated.append(item)
    return consolidated

single_core_tests = ['compress-single', 'fftw']

for d in [epyc_7b12_results,  epyc_7443_results, epyc_9b14_results]:
    for test in d.keys():
        if test != 'num_cores':
            d[test]['throughput'] = consolidate_list_sequences(d[test]['throughput'])
            if test not in single_core_tests:
                d[test]['power'] = d[test]['power'] / d['num_cores']
                d[test]['throughput'] = np.array(d[test]['throughput']) / d['num_cores']

def compute_normalized_throughput(throughput, reference_throughput, result_type):
    reference_throughput = np.array(reference_throughput)
    throughput = np.array(throughput)
    normalized_throughput = np.ones_like(throughput)
    for i in range(len(throughput)):
        if result_type[i] == 'higher':
            normalized_throughput[i] = throughput[i] / reference_throughput[i]
        else:
            normalized_throughput[i] = reference_throughput[i] / throughput[i]

    return np.nanmean(normalized_throughput)


tests = sorted(epyc_7b12_results.keys())
normalized_throghputs = {
    'EPYC 7B12': [compute_normalized_throughput(epyc_7b12_results[test]['throughput'], epyc_7b12_results[test]['throughput'], epyc_7b12_results[test]['result_type']) for test in tests if test !='num_cores'],
    'EPYC 7443': [compute_normalized_throughput(epyc_7443_results[test]['throughput'], epyc_7b12_results[test]['throughput'], epyc_7b12_results[test]['result_type']) for test in tests if test !='num_cores'],
    'EPYC 9B14': [compute_normalized_throughput(epyc_9b14_results[test]['throughput'], epyc_7b12_results[test]['throughput'], epyc_7b12_results[test]['result_type']) for test in tests if test !='num_cores'],
}
power = {
    'EPYC 7B12': [epyc_7b12_results[test]['power'] for test in tests if test !='num_cores'],
    'EPYC 7443': [epyc_7443_results[test]['power'] for test in tests if test !='num_cores'],
    'EPYC 9B14': [epyc_9b14_results[test]['power'] for test in tests if test !='num_cores'],
}

throughput_normalized_power = {
    'EPYC 7B12': np.array(power['EPYC 7B12']) / np.array(normalized_throghputs['EPYC 7B12']),
    'EPYC 7443': np.array(power['EPYC 7443']) / np.array(normalized_throghputs['EPYC 7443']),
    'EPYC 9B14': np.array(power['EPYC 9B14']) / np.array(normalized_throghputs['EPYC 9B14']),
}

tnp_to_7b12 = {
    'EPYC 7B12': [1.0]*len(throughput_normalized_power['EPYC 7B12']),
    'EPYC 7443': throughput_normalized_power['EPYC 7443'] / throughput_normalized_power['EPYC 7B12'],
    'EPYC 9B14': throughput_normalized_power['EPYC 9B14'] / throughput_normalized_power['EPYC 7B12'],
}

selected_tests = ['compile','compress','fft','npb','spark','openssl','x264']
normalized_throughputs = {'EPYC 7B12': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
 'EPYC 7443': [0.6405367000067509,
  1.2590169831930185,
  1.2113404313404312,
  1.3553737064870262,
  0.9616697870438536,
  1.1433920610347212,
  1.6669005512518118],
 'EPYC 9B14': [1.1607831717291734,
  1.1114177374207037,
  1.221119301119301,
  1.4005634192359444,
  1.3072675768106556,
  1.5489780033434224,
  1.4957393010187274]
}
normalized_energy_ratio = {
    'EPYC 7B12': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
    'EPYC 7443': [3.23957712,  1.30842027,  1.97799155,
       2.1565586 , 2.61230555, 1.70912891, 1.86206523], 
    'EPYC 9B14': [0.74636702,  0.75910529,  0.74236601,
       0.58284972, 0.76238793, 0.61605968, 0.67318546]
}

original_power = {
    'EPYC 7B12': [123.71,  46.7,  64.34, 76.48, 98.56, 136.84, 58.37], 
    'EPYC 7443': [192.53,  76.93,  115.62, 167.66, 185.7, 200.56, 135.88], 
    'EPYC 9B14': [100.48,  39.4,  54.68, 58.53, 92.09, 122.42, 55.1]
}

# system specs: 
# 1) GCP N2D 32 cores (1/2 7B12), 256GB RAM, 100GB HDD
# 2) Rightsized Local Server, 24 cores (1 7443), 256GB RAM, 100GB HDD
# 3) GCP C3D 30 cores (5/16 9B14), 256GB RAM, 100GB HDD

# Calculate impacts per year (embodied over 5 years + operational)
server_configs = {
   'EPYC 7B12': {
      'cpu': 'EPYC 7B12', 
      'ram': 256,
      'power_tests': original_power['EPYC 7B12'],
      'throughput': normalized_throughputs['EPYC 7B12'],
      'energy_ratio': normalized_energy_ratio['EPYC 7B12'],
      'production_year': 2019,
      'vm_ratio': 0.5
   },
   'EPYC 7443': {
      'cpu': 'EPYC 7443',
      'ram': 256,
      'power_tests': original_power['EPYC 7443'],
      'throughput': normalized_throughputs['EPYC 7443'],
      'energy_ratio': normalized_energy_ratio['EPYC 7443'],
      'production_year': 2021,
      'vm_ratio': 1.0
   },
   'EPYC 9B14': {
      'cpu': 'EPYC 9B14',
      'ram': 240, 
      'power_tests': original_power['EPYC 9B14'],
      'throughput': normalized_throughputs['EPYC 9B14'],
      'energy_ratio': normalized_energy_ratio['EPYC 9B14'],
      'production_year': 2023,
      'vm_ratio': 5/16
   }
}

# Calculate embodied and operational impacts for each server
server_impacts = {}

for server_name, config in server_configs.items():
   # Calculate embodied impacts
   if server_name == 'EPYC 7B12':
      cpu_specs = epyc_7b12_specs.copy()
   elif server_name == 'EPYC 7443':
      cpu_specs = epyc_7443_specs.copy()
   else:
      cpu_specs = epyc_9b14_specs.copy()
   
   # Calculate embodied impacts
   cpu_impact = calculate_total_impact(cpu_specs,server_configs[server_name]['vm_ratio'])
   
   # RAM impacts
   mem_specs = {
      'component_type': 'DRAM',
      'capacity': config['ram'],
      'production_year': config['production_year'],
      'distance': distance['default']
   }
   mem_impact = calculate_total_impact(mem_specs)
   
   test_op_impacts = []
   # Get emission factors for that year
   year_idx = 2023-2016
   sox_ef = unified_emission_factors['SOx']['MidWest'][year_idx]  # g/kWh
   nox_ef = unified_emission_factors['NOx']['MidWest'][year_idx]  # g/kWh 
   nh3_ef = unified_emission_factors['NH3']['US'][year_idx]  # g/kWh

   for i, test, power in zip(range(len(tests)), tests, config['power_tests']):
      power = power / config['throughput'][i] * 1.2
      annual_energy = power * 3600 * 24 * 365 / 1000  # kWh/year
      test_op_impacts.append({
         'test': test,
         'annual_energy': annual_energy,
         'impact': {
         'AP': (sox_ef * impact_factors['AP']['SOx'] + 
               nox_ef * impact_factors['AP']['NOx'] + 
               nh3_ef * impact_factors['AP']['NH3']) * annual_energy / 1000,
         'EP': (nox_ef * impact_factors['EP']['NOx'] + 
               nh3_ef * impact_factors['EP']['NH3']) * annual_energy / 1000,
         'FETox': 0  # Assuming negligible direct ecotoxicity impact
         }
   })
   
   # Calculate total impacts (embodied / 5 + operational)
   total_impacts = {
      'embodied': {
         'CPU': {k: v/5 for k, v in cpu_impact['total']['midpoint'].items()},
         'Memory': {k: v/5 for k, v in mem_impact['total']['midpoint'].items()}
      },
      'operational': test_op_impacts,
      'tests' : tests,
      'yearly_total': [{
         'AP': cpu_impact['total']['midpoint']['AP']/5 + mem_impact['total']['midpoint']['AP']/5 + op_impact['impact']['AP'] ,
         'EP': cpu_impact['total']['midpoint']['EP']/5 + mem_impact['total']['midpoint']['EP']/5 + op_impact['impact']['EP'] ,
         'FETox': cpu_impact['total']['midpoint']['FETox']/5 + mem_impact['total']['midpoint']['FETox']/5 + op_impact['impact']['FETox'] 
      }for op_impact in test_op_impacts]
   }
   
   # Convert to endpoint
   total_impacts['yearly_endpoint'] = [midpoint_to_endpoint(impact) for impact in total_impacts['yearly_total']]
   total_impacts['total_endpoint_value'] = [sum(impact.values()) for impact in total_impacts['yearly_endpoint']]

   
   server_impacts[server_name] = total_impacts

# Calculate normalized impacts using EPYC 7B12 as reference
reference_impact = np.array(server_impacts['EPYC 7B12']['total_endpoint_value'])

normalized_impacts = {
   'EPYC 7B12': np.ones_like(reference_impact),
   'EPYC 7443': np.array(server_impacts['EPYC 7443']['total_endpoint_value']) / reference_impact,
   'EPYC 9B14': np.array(server_impacts['EPYC 9B14']['total_endpoint_value']) / reference_impact
}