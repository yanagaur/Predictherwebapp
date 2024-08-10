import 'package:flutter/material.dart';

class DetailedQuestionnaireScreen extends StatefulWidget {
  final bool hasHormonalReports;

  DetailedQuestionnaireScreen({required this.hasHormonalReports});

  @override
  _DetailedQuestionnaireScreenState createState() =>
      _DetailedQuestionnaireScreenState();
}

class _DetailedQuestionnaireScreenState
    extends State<DetailedQuestionnaireScreen> {
  final GlobalKey<FormState> _formKey = GlobalKey<FormState>();

  // Controllers for text fields
  final Map<String, TextEditingController> _controllers = {
    'age': TextEditingController(),
    'weight': TextEditingController(),
    'height': TextEditingController(),
    'bmi': TextEditingController(),
    'bmiCategory': TextEditingController(),
    'bloodGroup': TextEditingController(),
    'pulseRate': TextEditingController(),
    'rr': TextEditingController(),
    'hb': TextEditingController(),
    'cycle': TextEditingController(),
    'cycleLength': TextEditingController(),
    'marriageStatus': TextEditingController(),
    'noOfAbortions': TextEditingController(),
    'iBetaHCG': TextEditingController(),
    'iiBetaHCG': TextEditingController(),
    'fsh': TextEditingController(),
    'lh': TextEditingController(),
    'fshLHRatio': TextEditingController(),
    'hip': TextEditingController(),
    'waist': TextEditingController(),
    'waistHipRatio': TextEditingController(),
    'tsh': TextEditingController(),
    'amh': TextEditingController(),
    'prl': TextEditingController(),
    'vitD3': TextEditingController(),
    'prg': TextEditingController(),
    'rbs': TextEditingController(),
    'bpSystolic': TextEditingController(),
    'bpDiastolic': TextEditingController(),
    'follicleNoL': TextEditingController(),
    'follicleNoR': TextEditingController(),
    'avgFSizeL': TextEditingController(),
    'avgFSizeR': TextEditingController(),
    'endometrium': TextEditingController(),
  };

  // State variables for Yes/No fields
  Map<String, String> _yesNoFields = {
    'pregnant': 'N',
    'weightGain': 'N',
    'hairGrowth': 'N',
    'skinDarkening': 'N',
    'hairLoss': 'N',
    'pimples': 'N',
    'fastFood': 'N',
    'regExercise': 'N',
  };

  @override
  void dispose() {
    // Dispose all controllers when screen is disposed
    _controllers.forEach((key, controller) {
      controller.dispose();
    });
    super.dispose();
  }

  void _handleSubmit() {
    if (_formKey.currentState!.validate()) {
      // Form is valid, proceed with data submission
      Map<String, dynamic> formData = {};

      // Collect data from controllers
      _controllers.forEach((key, controller) {
        formData[key] = controller.text;
      });

      // Add Yes/No fields to formData
      formData.addAll(_yesNoFields);

      // Calculate BMI if height and weight are provided
      if (_controllers['height']!.text.isNotEmpty &&
          _controllers['weight']!.text.isNotEmpty) {
        double height = double.parse(_controllers['height']!.text) / 100;
        double weight = double.parse(_controllers['weight']!.text);

        // Calculate BMI
        double bmi = weight / (height * height);

        // Determine BMI category
        String bmiCategory = _getBMICategory(bmi);

        // Add BMI and BMI category to formData
        formData['bmi'] = bmi.toStringAsFixed(1); // Store BMI with one decimal place
        formData['bmiCategory'] = bmiCategory;
      }

      // Print all collected data (for demonstration purposes)
      print(formData);

      // You can handle data submission here (e.g., send to server, save locally)
    }
  }

  String _getBMICategory(double bmi) {
    if (bmi < 18) {
      return 'Underweight';
    } else if (bmi >= 18 && bmi < 25) {
      return 'Normal weight';
    } else if (bmi >= 25 && bmi < 30) {
      return 'Overweight';
    } else {
      return 'Obese';
    }
  }

  Widget _buildYesNoButtons(
      String label, String stateVar, Function(String) onChanged) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        children: [
          Expanded(child: Text(label, style: TextStyle(fontSize: 16))),
          SizedBox(width: 10),
          ElevatedButton(
            onPressed: () => onChanged('Y'),
            child: Text('Yes'),
            style: ElevatedButton.styleFrom(
              backgroundColor: stateVar == 'Y' ? Colors.pinkAccent : Colors.grey[300],
              foregroundColor: stateVar == 'Y' ? Colors.white : Colors.black,
            ),
          ),
          SizedBox(width: 10),
          ElevatedButton(
            onPressed: () => onChanged('N'),
            child: Text('No'),
            style: ElevatedButton.styleFrom(
              backgroundColor: stateVar == 'N' ? Colors.pinkAccent : Colors.grey[300],
              foregroundColor: stateVar == 'N' ? Colors.white : Colors.black,
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Detailed Questionnaire'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SingleChildScrollView(
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: <Widget>[
                Text(
                  'Personal Information',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                Divider(),
                ..._buildSection([
                  'age',
                  'weight',
                  'height',
                  'bmi',
                  'bmiCategory',
                  'bloodGroup',
                ]),
                SizedBox(height: 20),
                Text(
                  'Health Metrics',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                Divider(),
                ..._buildSection([
                  'pulseRate',
                  'rr',
                  'hb',
                  'cycle',
                  'cycleLength',
                  'marriageStatus',
                  'noOfAbortions',
                ]),
                SizedBox(height: 20),
                Text(
                  'Hormonal Reports',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                Divider(),
                ..._buildSection([
                  'iBetaHCG',
                  'iiBetaHCG',
                  'fsh',
                  'lh',
                  'fshLHRatio',
                  'hip',
                  'waist',
                  'waistHipRatio',
                  'tsh',
                  'amh',
                  'prl',
                  'vitD3',
                  'prg',
                  'rbs',
                  'bpSystolic',
                  'bpDiastolic',
                  'follicleNoL',
                  'follicleNoR',
                  'avgFSizeL',
                  'avgFSizeR',
                  'endometrium',
                ]),
                SizedBox(height: 20),
                Text(
                  'Lifestyle',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                Divider(),
                ..._yesNoFields.entries.map((entry) {
                  String label = _getFormattedLabel(entry.key);
                  String stateVar = entry.value;
                  return _buildYesNoButtons(label, stateVar, (value) {
                    setState(() {
                      _yesNoFields[entry.key] = value;
                    });
                  });
                }).toList(),
                SizedBox(height: 20),
                ElevatedButton(
                  onPressed: _handleSubmit,
                  child: Text('Submit'),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  List<Widget> _buildSection(List<String> keys) {
    return keys.map((key) {
      String label = _getFormattedLabel(key);
      TextEditingController controller = _controllers[key]!;
      return Padding(
        padding: const EdgeInsets.symmetric(vertical: 8.0),
        child: TextFormField(
          controller: controller,
          decoration: InputDecoration(
            labelText: label,
            border: OutlineInputBorder(),
            filled: true,
            fillColor: Colors.grey[200],
            contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          ),
          keyboardType: _getKeyboardType(key),
          validator: (value) {
            if (value == null || value.isEmpty) {
              return '$label is required';
            }
            return null;
          },
        ),
      );
    }).toList();
  }

  String _getFormattedLabel(String key) {
    // Capitalize the first letter and add spaces before capital letters
    return key.replaceAllMapped(
      RegExp(r'([a-z])([A-Z])'),
          (match) => '${match.group(1)} ${match.group(2)}',
    ).replaceFirstMapped(
      RegExp(r'^[a-z]'),
          (match) => match.group(0)!.toUpperCase(),
    );
  }


  TextInputType _getKeyboardType(String key) {
    // Determine keyboardType based on key (customize as needed)
    if (key.contains('age') ||
        key.contains('weight') ||
        key.contains('height') ||
        key.contains('pulseRate') ||
        key.contains('rr') ||
        key.contains('hb') ||
        key.contains('cycleLength') ||
        key.contains('noOfAbortions') ||
        key.contains('iBetaHCG') ||
        key.contains('iiBetaHCG') ||
        key.contains('fsh') ||
        key.contains('lh') ||
        key.contains('hip') ||
        key.contains('waist') ||
        key.contains('tsh') ||
        key.contains('amh') ||
        key.contains('prl') ||
        key.contains('vitD3') ||
        key.contains('prg') ||
        key.contains('rbs') ||
        key.contains('bpSystolic') ||
        key.contains('bpDiastolic') ||
        key.contains('follicleNoL') ||
        key.contains('follicleNoR') ||
        key.contains('avgFSizeL') ||
        key.contains('avgFSizeR') ||
        key.contains('endometrium')) {
      return TextInputType.number;
    } else {
      return TextInputType.text;
    }
  }
}
