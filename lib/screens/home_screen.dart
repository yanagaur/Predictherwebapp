import 'package:flutter/material.dart';
import 'detailedquestionnairescreen_no.dart';

class HomeScreen extends StatefulWidget {
  final Function(bool) toggleTheme;
  final bool isDarkMode;

  HomeScreen({Key? key, required this.toggleTheme, required this.isDarkMode}) : super(key: key);

  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final GlobalKey<FormState> _formKey = GlobalKey<FormState>();

  void _handleSubmit() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => DetailedQuestionnaireScreen_no(hasHormonalReports: false),
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
              crossAxisAlignment: CrossAxisAlignment.center,
              children: <Widget>[
                SizedBox(height: 50),
                ClipRRect(
                  borderRadius: BorderRadius.circular(30),
                  child: Image.asset(
                    'assets/img_3.png', // Ensure this path is correct in your project
                    height: 320,
                    width: screenWidth * 0.9,
                    fit: BoxFit.cover,
                  ),
                ),
                SizedBox(height: 30),
                Text(
                  'Welcome to PredictHer',
                  style: TextStyle(
                    fontSize: 32,
                    fontWeight: FontWeight.bold,
                    color: Colors.pink,
                  ),
                  textAlign: TextAlign.center,
                ),
                SizedBox(height: 60),
                Text(
                  'Get your first insights on PCOD quickly and easily.',
                  style: TextStyle(
                    fontSize: 20,
                    color: Colors.grey[700],
                    fontStyle: FontStyle.italic,
                  ),
                  textAlign: TextAlign.center,
                ),
                SizedBox(height: 40),

                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 20.0),
                  child: Text(
                    '',
                    style: TextStyle(
                      fontSize: 18,
                      color: Colors.black87,
                      height: 1.5,
                    ),
                    textAlign: TextAlign.center,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _handleSubmit,
        backgroundColor: Colors.pink,
        child: Icon(
          Icons.arrow_forward, // Use an arrow icon
          color: Colors.white,
          size: 30,
        ),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat, // Centered at the bottom
    );
  }
}