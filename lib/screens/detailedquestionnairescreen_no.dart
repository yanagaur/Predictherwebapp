import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'prediction_result_screen.dart';

class DetailedQuestionnaireScreen_no extends StatefulWidget {
  final bool hasHormonalReports;

  DetailedQuestionnaireScreen_no({required this.hasHormonalReports});

  @override
  _DetailedQuestionnaireScreenState createState() => _DetailedQuestionnaireScreenState();
}

class _DetailedQuestionnaireScreenState extends State<DetailedQuestionnaireScreen_no> {
  final GlobalKey<FormState> _formKey = GlobalKey<FormState>();

  // Controllers for text fields
  final Map<String, TextEditingController> _controllers = {
    'age': TextEditingController(),
    'waist': TextEditingController(),
    'bmi': TextEditingController(),
    'cycleLength': TextEditingController(),
  };

  // State variables for Yes/No fields
  Map<String, String> _yesNoFields = {
    'weightGain': '',
    'hairGrowth': '',
    'skinDarkening': '',
    'hairLoss': '',
    'pimples': '',
    'fastFood': '',
    'regExercise': '',
  };

  String _cycleType = ''; // To store the selected cycle type ('Regular' or 'Irregular')

  @override
  void dispose() {
    // Dispose all controllers when screen is disposed
    _controllers.forEach((key, controller) {
      controller.dispose();
    });
    super.dispose();
  }

  Future<Map<String, dynamic>?> sendData(Map<String, dynamic> formData) async {
    try {
      print(json.encode(formData));

      final response = await http.post(
        Uri.parse('http://127.0.0.1:8989/noreport'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode(formData),
      );

      if (response.statusCode == 200) {
        // Handle successful response
        final responseData = json.decode(response.body);
        print('Response from server: $responseData');
        return responseData;
      } else {
        // Handle error response
        print('Error: ${response.statusCode}');
        return null;
      }
    } catch (e) {
      print('Error sending data: $e');
      return null;
    }
  }

  void _handleSubmit() async {
    if (_formKey.currentState!.validate()) {
      // Form is valid, proceed with data submission
      Map<String, dynamic> formData = {};

      // Collect data from controllers
      _controllers.forEach((key, controller) {
        formData[key] = controller.text;
      });

      // Add Yes/No fields to formData
      formData.addAll(_yesNoFields);

      // Add cycle type to formData
      formData['Cycle(R/I)'] = _cycleType;

      // Calculate BMI if waist is provided
      if (_controllers['waist']!.text.isNotEmpty) {
        try {
          double waist = double.parse(_controllers['waist']!.text);

          // Determine BMI category
          String bmiCategory = _getBMICategory(waist);

          // Add BMI and BMI category to formData
          formData['bmi'] = waist.toStringAsFixed(1); // Store BMI with one decimal place
          formData['bmiCategory'] = bmiCategory;

          // Set BMI category flags
          formData['BMI_a'] = bmiCategory == 'A' ? 1.0 : 0.0;
          formData['BMI_b'] = bmiCategory == 'B' ? 1.0 : 0.0;
          formData['BMI_c'] = bmiCategory == 'C' ? 1.0 : 0.0;
        } catch (e) {
          print('Error calculating BMI: $e');
        }
      }

      // Send data to the server
      try {
        final response = await sendData(formData);
        if (response != null) {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => PredictionResultScreen(
                predictions: response['predictions'],
                bmi_a: response['bmi_a'],
                bmi_b: response['bmi_b'],
                bmi_c: response['bmi_c'],
              ),
            ),
          );
        } else {
          _showErrorDialog('Error submitting data. Please try again later.');
        }
      } catch (e) {
        print('Error sending data: $e');
        _showErrorDialog('Error sending data: $e');
      }
    }
  }

  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Error'),
          content: Text(message),
          actions: <Widget>[
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
              },
              child: Text('OK'),
            ),
          ],
        );
      },
    );
  }

  String _getBMICategory(double bmi) {
    if (bmi < 18) {
      return 'A';
    } else if (bmi >= 18 && bmi < 26) {
      return 'B';
    } else {
      return 'C';
    }
  }

  Widget _buildYesNoButtons(String label, String stateVar, Function(String) onChanged) {
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
              foregroundColor: stateVar == 'Y' ? Colors.white : Colors.black,
              backgroundColor: stateVar == 'Y' ? Colors.pinkAccent : Colors.grey[300],
            ),
          ),
          SizedBox(width: 10),
          ElevatedButton(
            onPressed: () => onChanged('N'),
            child: Text('No'),
            style: ElevatedButton.styleFrom(
              foregroundColor: stateVar == 'N' ? Colors.white : Colors.black,
              backgroundColor: stateVar == 'N' ? Colors.pinkAccent : Colors.grey[300],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCycleButtons() {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Cycle',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: ElevatedButton(
                  onPressed: () {
                    setState(() {
                      _cycleType = 'Regular';
                    });
                  },
                  child: Text('Regular'),
                  style: ElevatedButton.styleFrom(
                    foregroundColor: _cycleType == 'Regular' ? Colors.white : Colors.black,
                    backgroundColor: _cycleType == 'Regular' ? Colors.pinkAccent : Colors.grey[300],
                  ),
                ),
              ),
              SizedBox(width: 10),
              Expanded(
                child: ElevatedButton(
                  onPressed: () {
                    setState(() {
                      _cycleType = 'Irregular';
                    });
                  },
                  child: Text('Irregular'),
                  style: ElevatedButton.styleFrom(
                    foregroundColor: _cycleType == 'Irregular' ? Colors.white : Colors.black,
                    backgroundColor: _cycleType == 'Irregular' ? Colors.pinkAccent : Colors.grey[300],
                  ),
                ),
              ),
            ],
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
                ..._buildSection(['age', 'waist', 'bmi']),
                SizedBox(height: 20),
                Text(
                  'Health Metrics',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
                Divider(),
                ..._buildSection(['Cycle length']),
                _buildCycleButtons(), // Add cycle buttons here
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
      TextEditingController? controller = _controllers[key];
      return Padding(
        padding: const EdgeInsets.symmetric(vertical: 8.0),
        child: TextFormField(
          controller: controller,
          keyboardType: key == 'age' ||
              key == 'waist' ||
              key == 'cycleLength' ||
              key == 'bmi'
              ? TextInputType.numberWithOptions(decimal: true)
              : TextInputType.text,
          inputFormatters: key == 'age' ||
              key == 'waist' ||
              key == 'cycleLength' ||
              key == 'bmi'
              ? [FilteringTextInputFormatter.allow(RegExp(r'^\d+\.?\d*'))]
              : [],
          validator: (value) {
            if (value == null || value.isEmpty) {
              return 'Please enter your $label';
            }
            return null;
          },
          decoration: InputDecoration(
            labelText: label,
            border: OutlineInputBorder(),
          ),
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
}
