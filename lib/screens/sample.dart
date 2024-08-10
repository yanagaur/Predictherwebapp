import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
// home
import 'detailedquestionnairescreen_no.dart';


class HomeScreen extends StatefulWidget {
  final Function(bool) toggleTheme;
  final bool isDarkMode;

  HomeScreen({Key? key, required this.toggleTheme, required this.isDarkMode}) : super(key: key);

  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final TextEditingController _cycleController = TextEditingController();
  final GlobalKey<FormState> _formKey = GlobalKey<FormState>();
  List<DateTime?> _selectedDates = [];
  int _numCycles = 0;

  @override
  void dispose() {
    _cycleController.dispose();
    super.dispose();
  }

  void _handleNext() {
    if (_formKey.currentState?.validate() ?? false) {
      final numCycles = int.tryParse(_cycleController.text) ?? 0;
      setState(() {
        _numCycles = numCycles;
        _selectedDates = List<DateTime?>.filled(_numCycles, null);
      });
      // Print the number of cycles
      // print('Number of Cycles: $_numCycles');
    }
  }

  Future<void> _selectDate(BuildContext context, int index) async {
    final DateTime? picked = await showDatePicker(
      context: context,
      initialDate: DateTime.now(),
      firstDate: DateTime(2000),
      lastDate: DateTime(2101),
      builder: (BuildContext context, Widget? child) {
        return Theme(
          data: Theme.of(context).copyWith(
            colorScheme: ColorScheme.light(
              primary: Colors.pink,
              onPrimary: Colors.white,
              onSurface: Colors.pinkAccent,
            ),
            textButtonTheme: TextButtonThemeData(
              style: TextButton.styleFrom(
                foregroundColor: Colors.pink,
              ),
            ),
          ),
          child: child!,
        );
      },
    );
    if (picked != null && picked != _selectedDates[index]) {
      setState(() {
        _selectedDates[index] = picked;
      });
    }
  }

  void _handleSubmitDates() {
    final dates = _selectedDates.map((date) => date != null ? DateFormat('yyyy-MM-dd').format(date!) : '').toList();

    // Print the selected dates
    //print('Selected Dates: $dates');

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => DetailedQuestionnaireScreen_no(hasHormonalReports: false,),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final double screenWidth = MediaQuery.of(context).size.width;

    return Scaffold(
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: <Widget>[
                SizedBox(
                  height: 50,
                ),
                Center(
                  child: ClipRRect(
                    borderRadius: BorderRadius.only(
                      topLeft: Radius.circular(30),
                      topRight: Radius.circular(30),
                    ),
                    child: Image.asset(
                      'assets/img_3.png',
                      height: 320,// Replace with your image path
                      width: screenWidth,
                      fit: BoxFit.cover,

                    ),
                  ),
                ),
                SizedBox(height: 1),
                Center(
                  child: Text(
                    'Cycle Tracker',
                    style: TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      color: Colors.pink,
                    ),
                  ),
                ),
                SizedBox(height: 20),
                TextFormField(
                  controller: _cycleController,
                  decoration: InputDecoration(
                    labelText: 'Number of Cycles',
                    labelStyle: TextStyle(color: Colors.pink),
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(10),
                      borderSide: BorderSide(color: Colors.pink),
                    ),
                    focusedBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(10),
                      borderSide: BorderSide(color: Colors.pink),
                    ),
                    prefixIcon: Icon(Icons.woman, color: Colors.pink),
                  ),
                  keyboardType: TextInputType.number,
                  validator: (value) {
                    if (value == null || value.isEmpty) {
                      return 'Please enter the number of cycles';
                    }
                    if (int.tryParse(value) == null || int.parse(value) <= 0) {
                      return 'Please enter a valid number';
                    }
                    return null;
                  },
                ),
                SizedBox(height: 20),
                ElevatedButton(
                  onPressed: _handleNext,
                  style: ElevatedButton.styleFrom(
                    foregroundColor: Colors.white,
                    backgroundColor: Colors.pink,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10),
                    ),
                    padding: EdgeInsets.symmetric(horizontal: 32, vertical: 12),
                  ),
                  child: Text('Next', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                ),
                if (_numCycles > 0) ...[
                  Divider(height: 40, thickness: 2, color: Colors.pinkAccent),
                  Text(
                    'Select Dates for Each Cycle',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.pink),
                  ),
                  SizedBox(height: 10),
                  ...List.generate(_numCycles, (index) {
                    return Card(
                      margin: const EdgeInsets.symmetric(vertical: 8.0),
                      elevation: 3,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10.0),
                      ),
                      child: ListTile(
                        title: Text(
                          _selectedDates[index] == null
                              ? 'Select Date for Cycle ${index + 1}'
                              : DateFormat('yyyy-MM-dd').format(_selectedDates[index]!),
                          style: TextStyle(fontSize: 16),
                        ),
                        trailing: Icon(Icons.calendar_today, color: Colors.pink),
                        onTap: () => _selectDate(context, index),
                      ),
                    );
                  }),
                  SizedBox(height: 20),
                  ElevatedButton(
                    onPressed: _handleSubmitDates,
                    style: ElevatedButton.styleFrom(
                      foregroundColor: Colors.white,
                      backgroundColor: Colors.pink,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                      padding: EdgeInsets.symmetric(horizontal: 32, vertical: 12),
                    ),
                    child: Text('Submit Dates', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}
